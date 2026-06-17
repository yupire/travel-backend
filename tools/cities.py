"""城市列表工具 — 接入和风天气 /geo/v2/city/lookup API。

该模块提供城市搜索功能，通过调用和风天气的城市查询接口获取真实数据��
- 冷启动时，使用预置的种子关键词批量调用城市查询接口
- 通过城市 id 去重，避免重复城市
- 结果在内存中缓存 24 小时，缓存跨越请求但在进程重启时清空
- 如果和风天气 API 失败（密钥缺失、网络错误等），抛出 QWeatherError 异常
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List

from tools.qweather import QWeatherError, _request, is_configured

log = logging.getLogger(__name__)

# 城市列表缓存有效期：24小时
# 城市数据相对稳定，长时间缓存可减少 API 调用次数
_CITIES_TTL_SECONDS = 24 * 60 * 60

# 城市查询种子关键词列表
# 这些关键词用于冷启动时批量调用城市查询接口，覆盖系统支持的主要城市
# 包含：国内热门城市、港澳台、日韩、东南亚、欧美、澳新、中东等地区
# 每次查询可能返回多个匹配结果（如查询"北京"会返回北京各区）
_SEED_QUERIES = [
    "北京", "上海", "广州", "深圳", "成都", "杭州", "西安", "重庆", "厦门", "青岛",
    "香港", "澳门", "台北",
    # "东京", "大阪", "京都", "札幌", "奈良", "横滨",
    # "首尔", "釜山",
    # "曼谷", "清迈", "普吉", "新加坡",
    # "巴黎", "伦敦", "罗马", "米兰", "巴塞罗那", "马德里", "柏林", "阿姆斯特丹",
    # "纽约", "洛杉矶", "旧金山", "芝加哥", "华盛顿",
    # "悉尼", "墨尔本", "奥克兰",
    # "迪拜", "伊斯坦布尔",
]

# 线程锁，用于保护城市列表缓存的并发访问
_lock = threading.Lock()
# 城市列表内存缓存（进程级），存储所有城市数据
_cities_cache: List[Dict] | None = None
# 缓存过期时间戳（使用单调时钟 time.monotonic()）
_cities_cache_expires_at: float = 0.0


def _fetch_all_cities() -> List[Dict]:
    """获取所有城市数据。

    通过遍历种子关键词，批量调用和风天气 /geo/v2/city/lookup 接口：
    1. 对每个种子关键词发起城市查询请求
    2. 通过城市 id 去重，避免重复城市
    3. 将 API 返回的字段映射为系统统一的城市数据格式
    4. 按国家、城市名排序，保证列表稳定性

    Returns:
        List[Dict]: 城市列表，每个城市包含 id/name/country/lat/lng 等字段

    Raises:
        QWeatherError: 当所有查询均失败时抛出
    """
    # 使用字典进行去重，key 为城市 id，value 为城市数据
    seen: Dict[str, Dict] = {}

    # 遍历所有种子关键词，逐个调用城市查询 API
    for q in _SEED_QUERIES:
        try:
            # 调用和风天气 /geo/v2/city/lookup 接口
            # location: 查询的城市名称
            # range: cn（国内）或 world（国际），根据是否包含中文字符自动判断
            # number: 每次查询最多返回 20 个结果
            # lang: 返回中文数据
            data = _request(
                "/geo/v2/city/lookup",
                params={"location": q, "range": "cn" if any("一" <= ch <= "鿿" for ch in q) else "world", "number": 20, "lang": "zh"},
                cache_ttl=_CITIES_TTL_SECONDS,  # 单个查询结果也缓存 24 小时
            )
        except QWeatherError as e:
            # 单个查询失败不中断整体流程，记录警告后继续
            log.warning("QWeather lookup %r 失败：%s", q, e)
            continue
        # 遍历 API 返回的城市列表，进行去重和字段映射
        for item in data.get("location", []):
            cid = item.get("id")
            # 跳过无效 id 或已存在的城市（去重）
            if not cid or cid in seen:
                continue
            # 仅保留城市类型的数据，过滤掉景区等其他类型
            if item.get("type") not in ("city", "scenic"):
                continue
            # 将 API 返回的字段映射为系统统一格式
            seen[cid] = {
                "id": cid,                      # 城市唯一标识
                "name": item.get("name", ""),   # 城市名称（中文）
                "name_en": item.get("name", ""),  # 城市英文名（API 暂未提供，复用 name）
                "country": item.get("country", ""),  # 国家
                "lat": float(item.get("lat", 0.0) or 0.0),  # 纬度
                "lng": float(item.get("lon", 0.0) or 0.0),  # 经度
                "adm1": item.get("adm1", ""),    # 一级行政区（省/州）
                "adm2": item.get("adm2", ""),    # 二级行政区（市）
                "fx_link": item.get("fxLink", ""),  # 和风天气详情页链接
            }
    # 如果所有查询都失败，抛出异常
    if not seen:
        raise QWeatherError(
            "城市列表为空：QWeather 全部 seed 查询均失败，请检查 .env 凭据或网络"
        )

    # 按国家 -> 城市名 -> 城市id 排序，保证列表稳定性
    # 按国家分组可以让前端下拉列表更易浏览
    return sorted(seen.values(), key=lambda c: (c["country"], c["name"], c["id"]))


def get_city(location: str) -> Dict:
    """根据城市名称查询单个城市数据。

    该方法直接调用和风天气 /geo/v2/city/lookup 接口，查询指定名称的城市。
    返回第一个匹配的城市数据，包含 id/name/country/lat/lng 等字段。
    如果查询失败或未找到匹配的城市，抛出 QWeatherError 异常。

    Args:
        location: 城市名称，如 "北京"、"Tokyo"、"New York"

    Returns:
        Dict: 城市数据，包含 id/name/country/lat/lng 等字段

    Raises:
        QWeatherError: 当查询失败或未找到匹配的城市时抛出
    """
    try:
        data = _request(
            "/geo/v2/city/lookup",
            params={"location": location, "range": "cn" if any("一" <= ch <= "鿿" for ch in location) else "world", "number": 1, "lang": "zh"},
            cache_ttl=0,  # 不使用缓存，实时查询
        )
    except QWeatherError as e:
        raise QWeatherError(f"查询城市 {location} 失败：{e}")

    locations = data.get("location", [])
    if not locations:
        raise QWeatherError(f"未找到匹配的城市：{location}")

    item = locations[0]
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "name_en": item.get("name"),  # API 暂未提供英文名，复用 name
        "country": item.get("country"),
        "lat": float(item.get("lat", 0.0) or 0.0),
        "lng": float(item.get("lon", 0.0) or 0.0),
        "adm1": item.get("adm1", ""),
        "adm2": item.get("adm2", ""),
        "fx_link": item.get("fxLink", ""),
    }

def get_cities() -> List[Dict]:
    """获取城市列表（带缓存）。

    该方法实现了两级缓存策略：
    1. 进程级内存缓存（24 小时有效期）
    2. API 级缓存（在 _fetch_all_cities 中通过 _request 实现）

    如果缓存未命中，会调用 _fetch_all_cities 从和风天气 API 获取数据。

    Returns:
        List[Dict]: 城市列表

    Raises:
        QWeatherError: 当和风天气未配置或 API 调用失败时抛出
    """
    global _cities_cache, _cities_cache_expires_at
    import time

    # 检查和风天气是否已配置（私钥是否存在）
    if not is_configured():
        raise QWeatherError(
            "QWeather 未配置 QWEATHER_PRIVATE_KEY。请在 backend/.env 设置 Ed25519 私钥。"
        )

    # 使用线程锁保护缓存读写，保证并发安全
    with _lock:
        # 检查缓存是否存在且未过期
        if _cities_cache is not None and time.monotonic() < _cities_cache_expires_at:
            return _cities_cache

        # 缓存未命中，记录日志并重新获取数据
        log.info("城市列表缓存未命中，调用 QWeather /geo/v2/city/lookup 重建（%d 个 seed）", len(_SEED_QUERIES))
        _cities_cache = _fetch_all_cities()
        # 更新缓存过期时间戳
        _cities_cache_expires_at = time.monotonic() + _CITIES_TTL_SECONDS
        return _cities_cache


def get_city_ids() -> List[str]:
    """获取所有城市 ID 列表。

    Returns:
        List[str]: 城市ID列表，如 ["101010100", "101020100", ...]
    """
    return [c["id"] for c in get_cities()]
