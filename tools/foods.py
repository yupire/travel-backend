"""美食查询工具 — 高德地图 POI API + Mock 降级

该模块提供美食查询功能，优先调用高德地图 /v5/place/text API，
失败时降级到 mock_data.json 的本地数据。

支持：
- 中国城市（含港澳）→ 高德真实数据
- 国际城市 → Mock 数据
- API Key 未配置/网络错误 → Mock 降级
"""
from __future__ import annotations

import json
import logging
import math
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

# 缓存时间：24 小时（美食数据相对稳定）
_FOODS_TTL_SECONDS = 24 * 60 * 60

# 高德 POI 类型编码：餐饮服务
# 参考: https://lbs.amap.com/api/webservice/guide/api/search#poi_type_code
_FOOD_TYPES = "050000|050101|050102|050103|050104|050105|050107|050108|050109|050110|050111|050112|050118"

# 加载 mock 数据
def _load_mock_foods() -> Dict[str, List[Dict]]:
    """加载 mock_data.json 中的美食数据"""
    try:
        with open(_DATA_PATH, encoding="utf-8") as f:
            return json.load(f)["foods"]
    except Exception as e:
        log.error("加载 mock_data.json 失败: %s", e)
        return {}


_MOCK_FOODS_DB = _load_mock_foods()


# ---------------------------------------------------------------------------
# 距离计算工具
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> int:
    """计算两点间的球面距离（米）

    Args:
        lat1, lng1: 第一个点的纬度、经度
        lat2, lng2: 第二个点的纬度、经度

    Returns:
        int: 距离（米）
    """
    R = 6371000  # 地球半径（米）
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lng / 2) ** 2
    )
    return int(R * 2 * math.asin(math.sqrt(a)))


# ---------------------------------------------------------------------------
# 高德 API 数据处理
# ---------------------------------------------------------------------------

def _amap_poi_to_food(poi: Dict) -> Dict:
    """将高德 POI 数据转换为系统美食格式

    Args:
        poi: 高德 API 返回的单个 POI 数据

    Returns:
        Dict: 符合 FoodRec 格式的美食数据
    """
    location = poi.get("location", "")
    lng, lat = parse_location(location) if location else (0.0, 0.0)

    # 从 type 字段提取菜系类型
    poi_type = poi.get("type", "")
    type_parts = [t.strip() for t in poi_type.split(";") if t.strip()]

    # 菜系推断：取 type 的第二个层级（如 "中餐厅;火锅" → "火锅"）
    cuisine = type_parts[1] if len(type_parts) > 1 else type_parts[0] if type_parts else "美食"

    return {
        "id": poi.get("id", ""),
        "name": poi.get("name", ""),
        "lat": lat,
        "lng": lng,
        "cuisine": cuisine,
        "price_range": "$$",  # 默认价格区间
        "rating": 4.0,  # 默认评分
    }


def _fetch_foods_from_amap(city: str, limit: int = 20) -> List[Dict]:
    """从高德 API 获取美食列表

    Args:
        city: 城市名称（中文）
        limit: 返回数量限制

    Returns:
        List[Dict]: 美食列表

    Raises:
        AMapError: API 调用失败时
    """
    # 规范化城市名称
    region = normalize_city_name(city)

    # 计算需要请求的页数（每页最多 20 条）
    page_size = 20
    pages = (limit + page_size - 1) // page_size

    all_foods = []
    for page in range(1, pages + 1):
        try:
            data = _request(
                "/v5/place/text",
                params={
                    "keywords": "",
                    "types": _FOOD_TYPES,
                    "region": region,
                    "offset": str(page_size),
                    "page": str(page),
                },
                cache_ttl=_FOODS_TTL_SECONDS,
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
                food = _amap_poi_to_food(poi)
                all_foods.append(food)
            except Exception as e:
                log.warning("转换 POI 失败 %s: %s", poi.get("name"), e)

        if len(all_foods) >= limit:
            break

    return all_foods[:limit]


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def get_top_foods(city: str, limit: int = 20) -> List[Dict]:
    """获取城市热门美食列表

    优先使用高德真实数据，失败时降级到 mock。

    Args:
        city: 城市名称（中文或拼音）
        limit: 返回数量限制

    Returns:
        List[Dict]: 美食列表
    """
    cache_key = f"foods:{city}:{limit}"

    # 检查缓存
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # 判断是否为中国城市
    if not is_chinese_city(city):
        log.info("非中国城市 %s，使用 mock 美食数据", city)
        foods = _MOCK_FOODS_DB.get(city.lower(), [])[:limit]
        _cache_set(cache_key, foods, _FOODS_TTL_SECONDS)
        return foods

    # 尝试从高德获取
    if is_configured():
        try:
            foods = _fetch_foods_from_amap(city, limit)
            if foods:
                _cache_set(cache_key, foods, _FOODS_TTL_SECONDS)
                return foods
            else:
                log.warning("高德 API 返回空结果，降级到 mock")
        except AMapError as e:
            log.warning("高德 API 获取失败: %s，降级到 mock", e)
    else:
        log.info("AMAP_KEY 未配置，使用 mock 美食数据")

    # 降级到 mock
    foods = _MOCK_FOODS_DB.get(city.lower(), [])[:limit]
    _cache_set(cache_key, foods, _FOODS_TTL_SECONDS)
    return foods


def get_nearby_foods(spot: Dict, city_foods: List[Dict], limit: int = 4) -> List[Dict]:
    """获取景点附近的美食推荐

    根据距离排序，返回最近的 N 个美食。

    Args:
        spot: 景点信息（需包含 lat/lng 字段）
        city_foods: 城市美食列表
        limit: 返回数量限制

    Returns:
        List[Dict]: 美食列表，每个包含距离信息
    """
    scored = [
        (f, _haversine_m(spot["lat"], spot["lng"], f["lat"], f["lng"]))
        for f in city_foods
    ]
    scored.sort(key=lambda x: x[1])

    result = []
    for food, dist in scored[:limit]:
        result.append({
            "id": food["id"],
            "name": food["name"],
            "cuisine": food["cuisine"],
            "price_range": food.get("price_range", "$$"),
            "rating": food.get("rating", 4.0),
            "distance_m": dist,
        })
    return result
