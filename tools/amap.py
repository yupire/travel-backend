"""高德地图 API 客户端 — 统一访问层 + 内存缓存

这是高德地图 API 的统一访问层，负责：
1. API Key 配置检查
2. HTTP 请求封装
3. 响应数据内存缓存
4. 错误处理与降级

配置通过环境变量读取：
- AMAP_KEY: 高德地图 API Key

API 文档: https://lbs.amap.com/api/webservice/guide/api/search
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import httpx

log = logging.getLogger(__name__)

# 高德 API 主机地址
_DEFAULT_HOST = "https://restapi.amap.com"

# HTTP 请求超时时间（秒）
_HTTP_TIMEOUT = 10.0

# 内存缓存条目: (value, expires_at_monotonic)
_CacheVal = Tuple[Any, float]
_cache: Dict[str, _CacheVal] = {}
_cache_lock = threading.Lock()

# ──────────────────── 限流 / 重试配置 ────────────────────
# 高德对单 key 有「每秒并发数 (CUQPS)」限制，个人认证 key 通常只有几次/秒。
# 超限时高德返回 infocode=10021 (CUQPS_HAS_EXCEEDED_THE_LIMIT)。
# 下面用「信号量限制并发 + 最小请求间隔」把瞬时 QPS 压到限制以下，
# 并对限流类错误做指数退避重试，避免把临时限流当成致命错误抛出。

# 最大并发请求数（同一时刻最多有几个高德请求在途）
_MAX_CONCURRENCY = int(os.getenv("AMAP_MAX_CONCURRENCY", "2"))
# 相邻两次请求的最小间隔（秒），进一步平滑突发流量
_MIN_REQUEST_INTERVAL = float(os.getenv("AMAP_MIN_INTERVAL", "0.35"))
# 命中限流后的最大重试次数
_MAX_RETRIES = int(os.getenv("AMAP_MAX_RETRIES", "3"))
# 退避基数（秒）：第 n 次重试等待 _BACKOFF_BASE * 2**(n-1)
_BACKOFF_BASE = float(os.getenv("AMAP_BACKOFF_BASE", "0.5"))
# 可重试的高德业务 infocode（限流 / 并发超限类，等待后通常可恢复）
_RETRYABLE_INFOCODES = {
    "10021",  # CUQPS_HAS_EXCEEDED_THE_LIMIT 每秒并发量超限
    "10019",  # CUQPS_HAS_EXCEEDED_THE_LIMIT（部分接口编码）
    "10029",  # CUQPS 限制（按分钟）
}

# 控制并发的信号量
_concurrency_sem = threading.Semaphore(_MAX_CONCURRENCY)
# 保护「上次请求时间」的锁，用于实现最小请求间隔
_rate_lock = threading.Lock()
_last_request_at = 0.0


def _throttle() -> None:
    """在真正发起 HTTP 请求前调用，保证相邻请求间隔 >= _MIN_REQUEST_INTERVAL。

    用一把全局锁串行化「读取上次时间 + 必要时 sleep + 更新时间」，
    使得无论多少线程并发，发往高德的请求都被拉开到最小间隔以上。
    """
    global _last_request_at
    if _MIN_REQUEST_INTERVAL <= 0:
        return
    with _rate_lock:
        now = time.monotonic()
        wait = _last_request_at + _MIN_REQUEST_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _last_request_at = now


class AMapError(RuntimeError):
    """高德 API 错误异常类

    当以下情况发生时抛出：
    - 网络请求失败
    - HTTP 状态码非 200
    - JSON 解析失败
    - 业务 status 非 1（高德返回的成功状态码）
    - API Key 未配置
    """


# ---------------------------------------------------------------------------
# 配置读取 (Configuration)
# ---------------------------------------------------------------------------

def _api_key() -> str:
    """从环境变量读取高德 API Key

    优先使用 AMAP_API_KEY，兼容 AMAP_KEY
    """
    return os.getenv("AMAP_API_KEY", "") or os.getenv("AMAP_KEY", "")


def is_configured() -> bool:
    """检查高德 API Key 是否已配置

    Returns:
        bool: API Key 非空返回 True，否则返回 False
    """
    return bool(_api_key())


def host() -> str:
    """获取高德 API 主机地址"""
    return os.getenv("AMAP_HOST", _DEFAULT_HOST).rstrip("/")


# ---------------------------------------------------------------------------
# 内存缓存 (Caching)
# ---------------------------------------------------------------------------

def _cache_get(key: str) -> Optional[Any]:
    """从内存缓存获取数据

    如果数据已过期，自动删除并返回 None。

    Args:
        key: 缓存键

    Returns:
        缓存的值，未命中或过期返回 None
    """
    with _cache_lock:
        entry = _cache.get(key)
        if not entry:
            return None
        value, expires_at = entry
        if time.monotonic() >= expires_at:
            _cache.pop(key, None)
            return None
        return value


def _cache_set(key: str, value: Any, ttl_seconds: float) -> None:
    """设置内存缓存

    Args:
        key: 缓存键
        value: 要缓存的值
        ttl_seconds: 过期时间（秒）
    """
    with _cache_lock:
        _cache[key] = (value, time.monotonic() + ttl_seconds)


# ---------------------------------------------------------------------------
# HTTP 请求封装 (HTTP request)
# ---------------------------------------------------------------------------

def _request(
    path: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    cache_ttl: float = 0.0,
) -> Dict[str, Any]:
    """向高德 API 发起 GET 请求

    自动添加 API Key，处理错误，支持缓存。

    Args:
        path: API 路径，如 "/v5/place/text"
        params: URL 查询参数（key 会自动添加）
        cache_ttl: 缓存时间（秒），0 表示不缓存

    Returns:
        Dict: 解析后的 JSON 响应

    Raises:
        AMapError: 网络错误、HTTP 错误、JSON 解析错误、业务错误
    """
    if not is_configured():
        raise AMapError("AMAP_API_KEY 未配置。请在 backend/.env 设置 AMAP_API_KEY。")

    # 使用 (path, 参数) 作为缓存键，确保相同请求复用缓存
    cache_key = f"{path}?{frozenset((params or {}).items())}"
    if cache_ttl > 0:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    url = f"{host()}{path}"
    # 添加 API Key 参数
    all_params = {**(params or {}), "key": _api_key()}

    # 统一打印本次请求的入参（key 不打印），方便在出错时定位是哪些参数惹的祸。
    # 放在 _request 层，所有高德接口（路径规划 / POI 搜索等）都自动有入参日志。
    log.info("高德 API 请求 %s 入参=%s", path, params or {})

    # 命中限流 (infocode in _RETRYABLE_INFOCODES) 时退避重试；
    # attempt 从 0 开始，0 为首次请求，1.._MAX_RETRIES 为重试。
    last_error: Optional[AMapError] = None
    for attempt in range(_MAX_RETRIES + 1):
        # 信号量限制并发 + 最小间隔，把瞬时 QPS 压到高德限制以下
        with _concurrency_sem:
            _throttle()
            try:
                with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
                    resp = client.get(url, params=all_params)
            except httpx.HTTPError as e:
                raise AMapError(f"高德 API 网络错误 {path}: {e}") from e

        if resp.status_code != 200:
            raise AMapError(
                f"高德 API HTTP {resp.status_code} {path}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except ValueError as e:
            raise AMapError(f"高德 API 响应不是合法 JSON {path}: {e}") from e

        # 高德业务状态码：1 表示成功，0 表示失败
        status = data.get("status")
        if status == "1":
            if cache_ttl > 0:
                _cache_set(cache_key, data, cache_ttl)
            return data

        infocode = str(data.get("infocode", ""))
        info = data.get("info", "")

        # 限流类错误：还有重试机会就退避后重试，否则抛出
        if infocode in _RETRYABLE_INFOCODES and attempt < _MAX_RETRIES:
            backoff = _BACKOFF_BASE * (2 ** attempt)
            log.warning(
                "高德 API 限流 %s: infocode=%s, info=%s；第 %d/%d 次退避重试，等待 %.2fs",
                path, infocode, info, attempt + 1, _MAX_RETRIES, backoff,
            )
            last_error = AMapError(
                f"高德 API 业务错误 {path}: status={status}, infocode={infocode}, info={info}"
            )
            time.sleep(backoff)
            continue

        # 非限流错误，或重试已用尽：直接抛出（带上入参，便于定位）
        raise AMapError(
            f"高德 API 业务错误 {path}: status={status}, infocode={infocode}, "
            f"info={info}, 入参={params or {}}"
        )

    # 理论上不会走到这里（循环内要么 return 要么 raise），兜底抛最后一次错误
    raise last_error or AMapError(f"高德 API 请求失败 {path}: 未知错误")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def is_chinese_city(city_name: str) -> bool:
    """判断是否为中国城市（含港澳）

    基于城市名称白名单判断，适用于 MVP 场景。

    Args:
        city_name: 城市名称（中文或拼音）

    Returns:
        bool: 中国城市返回 True
    """
    # 常见中国城市白名单（中文名称）
    chinese_names = {
        "北京", "上海", "广州", "深圳", "成都", "杭州", "西安", "重庆",
        "厦门", "青岛", "武汉", "南京", "天津", "苏州", "长沙", "郑州",
        "大连", "宁波", "无锡", "沈阳", "哈尔滨", "济南", "昆明", "贵阳",
        "南宁", "南昌", "合肥", "太原", "石家庄", "呼和浩特", "长春",
        "兰州", "银川", "西宁", "乌鲁木齐", "拉萨", "海口", "三亚",
        "香港", "澳门", "台北", "高雄",
    }

    # 常见中国城市白名单（拼音）
    chinese_pinyin = {
        "beijing", "shanghai", "guangzhou", "shenzhen", "chengdu", "hangzhou",
        "xian", "chongqing", "xiamen", "qingdao", "wuhan", "nanjing", "tianjin",
        "suzhou", "changsha", "zhengzhou", "dalian", "ningbo", "wuxi", "shenyang",
        "harbin", "jinan", "kunming", "guiyang", "nanning", "nanchang", "hefei",
        "taiyuan", "shijiazhuang", "hohhot", "changchun", "lanzhou", "yinchuan",
        "xining", "urumqi", "lhasa", "haikou", "sanya",
        "hongkong", "macau", "taipei", "kaohsiung",
    }

    normalized = city_name.strip()
    return normalized in chinese_names or normalized.lower() in chinese_pinyin


# 行政区（adcode）数据非常稳定，缓存 7 天
_DISTRICT_TTL_SECONDS = 7 * 24 * 60 * 60


def city_to_adcode(city: str) -> Optional[str]:
    """城市名 → 高德 adcode（citycode）

    高德 v5 公交路径规划 `/v5/direction/transit/integrated` 的 city1/city2
    只接受 citycode（行政区编码 adcode），不接受城市名；传名字会报
    INVALID_PARAMS (infocode=20000)。这里用「行政区查询」接口把城市名解析成 adcode。

    解析策略：
    - 已是纯数字 → 视为 adcode 直接返回
    - 拼音 → 先 normalize_city_name 转中文，提升匹配率
    - 解析失败 / 非中国城市 → 返回 None（由调用方决定如何降级）

    结果带 7 天缓存（行政区数据稳定）。

    Args:
        city: 城市名（中文或拼音）或 adcode

    Returns:
        Optional[str]: adcode 字符串，解析不到返回 None
    """
    if not city:
        return None

    name = city.strip()
    # 已经是 adcode（纯数字），直接用
    if name.isdigit():
        return name

    # 拼音 → 中文，提升行政区接口匹配率
    normalized = normalize_city_name(name)
    try:
        data = _request(
            "/v3/config/district",
            params={"keywords": normalized, "subdistrict": "0"},
            cache_ttl=_DISTRICT_TTL_SECONDS,
        )
    except AMapError as e:
        log.warning("解析城市 adcode 失败 city=%s（归一化=%s）: %s", city, normalized, e)
        return None

    districts = data.get("districts") or []
    if not districts:
        log.warning("行政区查询无结果 city=%s（归一化=%s）", city, normalized)
        return None

    adcode = districts[0].get("adcode") or None
    log.info("城市 adcode 解析: %s -> %s (%s)",
             city, adcode, districts[0].get("name", ""))
    return adcode


def normalize_city_name(city: str) -> str:
    """将城市名称规范化为高德 API 接受的格式

    Args:
        city: 城市名称（可能是拼音或中文）

    Returns:
        str: 规范化后的城市名称（中文格式）
    """
    # 常见城市拼音 → 中文映射
    pinyin_to_chinese = {
        "beijing": "北京市",
        "shanghai": "上海市",
        "guangzhou": "广州市",
        "shenzhen": "深圳市",
        "chengdu": "成都市",
        "hangzhou": "杭州市",
        "xian": "西安市",
        "chongqing": "重庆市",
        "xiamen": "厦门市",
        "qingdao": "青岛市",
        "wuhan": "武汉市",
        "nanjing": "南京市",
        "tianjin": "天津市",
        "suzhou": "苏州市",
        "changsha": "长沙市",
        "zhengzhou": "郑州市",
        "dalian": "大连市",
        "ningbo": "宁波市",
        "wuxi": "无锡市",
        "shenyang": "沈阳市",
        "harbin": "哈尔滨市",
        "jinan": "济南市",
        "kunming": "昆明市",
        "guiyang": "贵阳市",
        "nanning": "南宁市",
        "nanchang": "南昌市",
        "hefei": "合肥市",
        "taiyuan": "太原市",
        "shijiazhuang": "石家庄市",
        "hohhot": "呼和浩特市",
        "changchun": "长春市",
        "lanzhou": "兰州市",
        "yinchuan": "银川市",
        "xining": "西宁市",
        "urumqi": "乌鲁木齐市",
        "lhasa": "拉萨市",
        "haikou": "海口市",
        "sanya": "三亚市",
        "hongkong": "香港",
        "macau": "澳门",
        "taipei": "台北市",
        "kaohsiung": "高雄市",
    }

    normalized = city.strip()
    # 如果已经是中文，直接返回（可能带"市"后缀）
    if any("一" <= ch <= "鿿" for ch in normalized):
        # 如果没有"市"后缀，补充上（高德 API 更倾向于带"市"）
        if not normalized.endswith("市") and not normalized.endswith("香港") and not normalized.endswith("澳门"):
            return normalized + "市"
        return normalized

    # 拼音映射
    chinese = pinyin_to_chinese.get(normalized.lower())
    if chinese:
        return chinese

    # 兜底：直接返回原值
    return normalized


def parse_location(location: str) -> tuple[float, float]:
    """解析高德 location 字段 (经度,纬度)

    Args:
        location: 高德返回的 location 字符串，如 "116.275179,39.999617"

    Returns:
        tuple[float, float]: (lng, lat) — 注意高德返回的是 (经度,纬度)
    """
    try:
        lng_str, lat_str = location.split(",")
        return (float(lng_str), float(lat_str))
    except (ValueError, AttributeError) as e:
        raise AMapError(f"无法解析 location 字段: {location}") from e
