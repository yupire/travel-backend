"""景点查询工具 — 高德地图 POI API + Mock 降级

该模块提供景点查询功能，优先调用高德地图 /v5/place/text API，
失败时降级到 mock_data.json 的本地数据。

支持：
- 中国城市（含港澳）→ 高德真实数据
- 国际城市 → Mock 数据
- API Key 未配置/网络错误 → Mock 降级
"""
from __future__ import annotations

import json
import logging
import os
import importlib.util
from typing import Dict, List

# 从 amap 模块导入（使用 try/except 避免触发 tools/__init__.py）
try:
    from tools.amap import (
        AMapError,
        _cache_get,
        _cache_set,
        _request,
        is_chinese_city,
        is_configured,
        normalize_city_name,
        parse_location,
    )
except ImportError:
    # 如果 tools/__init__.py 加载失败，直接导入 amap 模块
    import importlib.util
    spec = importlib.util.spec_from_file_location("amap", os.path.join(os.path.dirname(__file__), "amap.py"))
    amap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(amap)
    AMapError = amap.AMapError
    _cache_get = amap._cache_get
    _cache_set = amap._cache_set
    _request = amap._request
    is_chinese_city = amap.is_chinese_city
    is_configured = amap.is_configured
    normalize_city_name = amap.normalize_city_name
    parse_location = amap.parse_location

log = logging.getLogger(__name__)

# Mock 数据路径
_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/mock_data.json")

# 缓存时间：24 小时（景点数据相对稳定）
_SPOTS_TTL_SECONDS = 24 * 60 * 60

# 高德 POI 类型编码：风景名胜
# 参考: https://lbs.amap.com/api/webservice/guide/api/search#poi_type_code
_SCENIC_TYPES = "110000|110101|110102|110103|110104|110105|110106|110107|110108|110109|110110|110111|110112|110113|110114|110115"

# 室内景点 typecode 列表（用于判断 indoor/outdoor）
_INDOOR_TYPECODES = {
    "110103",  # 博物馆
    "110104",  # 宗教场所（寺庙/教堂等，部分室内）
    "110108",  # 纪念馆
    "110110",  # 特色街区/商业街
    "110111",  # 历史��筑
    "110114",  # 文化场馆
    "110115",  # 展览馆
}

# 加载 mock 数据
def _load_mock_spots() -> Dict[str, List[Dict]]:
    """加载 mock_data.json 中的景点数据"""
    try:
        with open(_DATA_PATH, encoding="utf-8") as f:
            return json.load(f)["spots"]
    except Exception as e:
        log.error("加载 mock_data.json 失败: %s", e)
        return {}


_MOCK_SPOTS_DB = _load_mock_spots()


# ---------------------------------------------------------------------------
# 高德 API 数据处理
# ---------------------------------------------------------------------------

def _amap_poi_to_spot(poi: Dict) -> Dict:
    """将高德 POI 数据转换为系统景点格式

    Args:
        poi: 高德 API 返回的单个 POI 数据

    Returns:
        Dict: 符合 SpotPlan 格式的景点数据
    """
    location = poi.get("location", "")
    lng, lat = parse_location(location) if location else (0.0, 0.0)

    typecode = poi.get("typecode", "")
    # 判断是否为室内景点
    is_indoor = typecode in _INDOOR_TYPECODES or typecode.startswith("11") and typecode in {
        "110103", "110108", "110110", "110111", "110114", "110115"
    }

    # 从 type 字段提取 tags（分号分隔）
    poi_type = poi.get("type", "")
    tags = [t.strip() for t in poi_type.split(";") if t.strip()] if poi_type else []

    # 根据类型推断推荐游玩时长（分钟）
    duration_map = {
        "110101": 120,  # 风景名胜区
        "110102": 90,   # 古迹
        "110103": 120,  # 博物馆
        "110104": 60,   # 宗教场所
        "110105": 90,   # 公园
        "110106": 180,  # 动物园/植物园
        "110107": 120,  # 水族馆
        "110108": 90,   # 纪念���
        "110109": 180,  # 游乐园
    }
    type_prefix = typecode[:6] if len(typecode) >= 6 else typecode
    duration_min = duration_map.get(type_prefix, 90)

    return {
        "id": poi.get("id", ""),
        "name": poi.get("name", ""),
        "lat": lat,
        "lng": lng,
        "duration_min": duration_min,
        "open_time": "09:00-18:00",  # 默认开放时间
        "ticket": 0,  # 默认免费（后续可通过详情接口获取）
        "type": "indoor" if is_indoor else "outdoor",
        "tags": tags,
        "description": "",  # 空描述，后续可接入详情接口
        "is_indoor": is_indoor,
    }


def _fetch_spots_from_amap(city: str, limit: int = 20) -> List[Dict]:
    """从高德 API 获取景点列表

    Args:
        city: 城市名称（中文）
        limit: 返回数量限制

    Returns:
        List[Dict]: 景点列表

    Raises:
        AMapError: API 调用失败时
    """
    # 规范化城市名称
    region = normalize_city_name(city)

    # 计算需要请求的页数（每页最多 20 条）
    page_size = 20
    pages = (limit + page_size - 1) // page_size

    all_spots = []
    for page in range(1, pages + 1):
        try:
            data = _request(
                "/v5/place/text",
                params={
                    "keywords": "",
                    "types": _SCENIC_TYPES,
                    "region": region,
                    "offset": str(page_size),
                    "page": str(page),
                },
                cache_ttl=_SPOTS_TTL_SECONDS,
            )
        except AMapError as e:
            if page == 1:
                # 第一页就失败，直接抛出异常
                raise
            # 后续页失败时，返回已获取的数据
            log.warning("高德 API 第 %d 页获取失败: %s", page, e)
            break

        pois = data.get("pois", [])
        if not pois:
            break

        # 转换 POI 数据
        for poi in pois:
            try:
                spot = _amap_poi_to_spot(poi)
                all_spots.append(spot)
            except Exception as e:
                log.warning("转换 POI 失败 %s: %s", poi.get("name"), e)

        if len(all_spots) >= limit:
            break

    return all_spots[:limit]


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def get_top_spots(city: str, limit: int = 20) -> List[Dict]:
    """获取城市热门景点列表

    优先使用高德真实数据，失败时降级到 mock。

    Args:
        city: 城市名称（中文或拼音）
        limit: 返回数量限制

    Returns:
        List[Dict]: 景点列表
    """
    cache_key = f"spots:{city}:{limit}"

    # 检查缓存
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # 判断是否为中国城市
    if not is_chinese_city(city):
        log.info("非中国城市 %s，使用 mock 景点数据", city)
        spots = _MOCK_SPOTS_DB.get(city.lower(), [])[:limit]
        _cache_set(cache_key, spots, _SPOTS_TTL_SECONDS)
        return spots

    # 尝试从高德获取
    if is_configured():
        try:
            spots = _fetch_spots_from_amap(city, limit)
            if spots:
                _cache_set(cache_key, spots, _SPOTS_TTL_SECONDS)
                return spots
            else:
                log.warning("高德 API 返回空结果，降级到 mock")
        except AMapError as e:
            log.warning("高德 API 获取失败: %s，降级到 mock", e)
    else:
        log.info("AMAP_KEY 未配置，使用 mock 景点数据")

    # 降级到 mock
    spots = _MOCK_SPOTS_DB.get(city.lower(), [])[:limit]
    # 为 mock 数据添加 is_indoor 字段
    for spot in spots:
        spot["is_indoor"] = spot.get("type") == "indoor"
    _cache_set(cache_key, spots, _SPOTS_TTL_SECONDS)
    return spots


def get_spot_map(city: str) -> Dict[str, Dict]:
    """获取城市景点字典 {spot_id: spot_dict}

    Args:
        city: 城市名称

    Returns:
        Dict[str, Dict]: 景点字典
    """
    return {s["id"]: s for s in get_top_spots(city)}


def geocode_spots(city: str, spot_names: List[str]) -> List[Dict]:
    """批量地理编码景点名称

    对于给定的景点名称列表，返回对应的景点信息（坐标等）。
    名称未找到时跳过（不报错），以便 LLM 自由输入不会中断规划。

    优先使用高德 API，失败时降级到 mock。

    Args:
        city: 城市名称
        spot_names: 景点名称列表

    Returns:
        List[Dict]: 景点信息列表 [{id, name, lat, lng, is_indoor}, ...]
    """
    if not spot_names:
        return []

    # 规范化城市名称
    normalized_city = normalize_city_name(city)

    # 如果不是中国城市，直接从 mock 查找
    if not is_chinese_city(city):
        spot_map = get_spot_map(city)
        results = []
        for name in spot_names:
            match = next(
                (s for s in spot_map.values() if s["name"] == name or s.get("name_en") == name),
                None,
            )
            if match:
                results.append({
                    "id": match["id"],
                    "name": match["name"],
                    "lat": match["lat"],
                    "lng": match["lng"],
                    "is_indoor": match.get("type") == "indoor",
                })
        return results

    # 尝试从高德批量查询
    if is_configured():
        try:
            all_spots = []
            # 分批查询（高德 API 可能不支持单次查询太多关键词）
            for name in spot_names:
                try:
                    data = _request(
                        "/v5/place/text",
                        params={
                            "keywords": name,
                            "types": _SCENIC_TYPES,
                            "region": normalized_city,
                            "offset": "1",
                            "page": "1",
                        },
                        cache_ttl=_SPOTS_TTL_SECONDS,
                    )
                    pois = data.get("pois", [])
                    if pois:
                        poi = pois[0]  # 取第一个结果
                        location = poi.get("location", "")
                        lng, lat = parse_location(location) if location else (0.0, 0.0)
                        typecode = poi.get("typecode", "")
                        is_indoor = typecode in _INDOOR_TYPECODES
                        all_spots.append({
                            "id": poi.get("id", ""),
                            "name": poi.get("name", ""),
                            "lat": lat,
                            "lng": lng,
                            "is_indoor": is_indoor,
                        })
                except AMapError as e:
                    log.warning("查询景点 %s 失败: %s", name, e)

            if all_spots:
                return all_spots
        except AMapError as e:
            log.warning("高德批量查询失败: %s，降级到 mock", e)

    # 降级到 mock
    spot_map = get_spot_map(city)
    results = []
    for name in spot_names:
        match = next(
            (s for s in spot_map.values() if s["name"] == name or s.get("name_en") == name),
            None,
        )
        if match:
            results.append({
                "id": match["id"],
                "name": match["name"],
                "lat": match["lat"],
                "lng": match["lng"],
                "is_indoor": match.get("type") == "indoor" or match.get("is_indoor", False),
            })
    return results
