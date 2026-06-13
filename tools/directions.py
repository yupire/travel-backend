"""路径规划工具 — 高德地图方向 API

该模块提供多种交通方式的路径规划功能，基于高德地图 Direction API v5。

支持：
- 驾车规划 (driving)
- 步行规划 (walking)
- 骑行规划 (bicycling)
- 电动车规划 (electrobike)
- 公交车规划 (transit/integrated)

API 文档: https://lbs.amap.com/api/webservice/guide/api/direction
"""
from __future__ import annotations

import logging
import os
import importlib.util
from typing import Dict, List, Optional, Union

# 从 amap 模块导入（使用 try/except 避免触发 tools/__init__.py）
try:
    from tools.amap import (
        AMapError,
        _cache_get,
        _cache_set,
        _request,
        is_configured,
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
    is_configured = amap.is_configured
    parse_location = amap.parse_location

log = logging.getLogger(__name__)

# 缓存时间：1 小时（路径数据相对稳定）
_DIRECTION_TTL_SECONDS = 60 * 60

# ---------------------------------------------------------------------------
# 高德 API 数据处理
# ---------------------------------------------------------------------------


def _parse_step(step: Dict) -> Dict:
    """解析单个路径步骤

    Args:
        step: 高德 API 返回的单个步骤数据

    Returns:
        Dict: 解析后的步骤数据
    """
    return {
        "instruction": step.get("instruction", ""),
        "orientation": step.get("orientation", ""),
        "road_name": step.get("road_name", ""),
        "distance": int(step.get("step_distance", 0)),
    }


def _parse_path(path: Dict, mode: str) -> Dict:
    """解析单条路径

    Args:
        path: 高德 API 返回的单条路径数据
        mode: 交通方式

    Returns:
        Dict: 解析后的路径数据
    """
    result = {
        "distance": int(path.get("distance", 0)),
        "steps": [],
    }

    # 根据不同交通方式解析不同字段
    if mode == "walking":
        cost = path.get("cost", {})
        result["duration"] = int(cost.get("duration", 0))
    elif mode in ("bicycling", "electrobike"):
        result["duration"] = int(path.get("duration", 0))
    elif mode == "driving":
        result["restriction"] = int(path.get("restriction", 0))

    # 解析步骤
    steps = path.get("steps", [])
    if steps:
        result["steps"] = [_parse_step(step) for step in steps]

    return result


def _format_location(location: str) -> str:
    """格式化经纬度字符串

    Args:
        location: 经纬度字符串 "lng,lat"

    Returns:
        str: 格式化后的经纬度字符串（保留6位小数）
    """
    try:
        lng, lat = parse_location(location)
        return f"{lng:.6f},{lat:.6f}"
    except AMapError:
        return location


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


def plan_driving_route(
    origin: str,
    destination: str,
) -> Dict:
    """驾车路径规划

    Args:
        origin: 起点经纬度，格式 "lng,lat" (经度在前，纬度在后)
        destination: 目的地经纬度，格式 "lng,lat"

    Returns:
        Dict: 路径规划结果，包含：
            - origin: 起点坐标
            - destination: 目的地坐标
            - taxi_cost: 预估打车费用（元）
            - paths: 路径列表，每条路径包含 distance(米) 和 steps

    Raises:
        AMapError: API 调用失败时
    """
    cache_key = f"driving:{origin}:{destination}"

    # 检查缓存
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    if not is_configured():
        raise AMapError("AMAP_API_KEY 未配置，无法使用路径规划功能")

    # 格式化经纬度
    formatted_origin = _format_location(origin)
    formatted_destination = _format_location(destination)

    # 请求高德 API
    data = _request(
        "/v5/direction/driving",
        params={
            "origin": formatted_origin,
            "destination": formatted_destination,
        },
        cache_ttl=_DIRECTION_TTL_SECONDS,
    )

    route = data.get("route", {})

    # 解析路径
    paths = []
    for path in route.get("paths", []):
        parsed_path = _parse_path(path, "driving")
        paths.append(parsed_path)

    result = {
        "origin": route.get("origin", formatted_origin),
        "destination": route.get("destination", formatted_destination),
        "taxi_cost": int(route.get("taxi_cost", 0)),
        "paths": paths,
    }

    _cache_set(cache_key, result, _DIRECTION_TTL_SECONDS)
    return result


def plan_walking_route(
    origin: str,
    destination: str,
) -> Dict:
    """步行路径规划

    Args:
        origin: 起点经纬度，格式 "lng,lat" (经度在前，纬度在后)
        destination: 目的地经纬度，格式 "lng,lat"

    Returns:
        Dict: 路径规划结果，包含：
            - origin: 起点坐标
            - destination: 目的地坐标
            - paths: 路径列表，每条路径包含 distance(米) 和 duration(秒)

    Raises:
        AMapError: API 调用失败时
    """
    cache_key = f"walking:{origin}:{destination}"

    # 检查缓存
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    if not is_configured():
        raise AMapError("AMAP_API_KEY 未配置，无法使用路径规划功能")

    # 格式化经纬度
    formatted_origin = _format_location(origin)
    formatted_destination = _format_location(destination)

    # 请求高德 API
    data = _request(
        "/v5/direction/walking",
        params={
            "origin": formatted_origin,
            "destination": formatted_destination,
        },
        cache_ttl=_DIRECTION_TTL_SECONDS,
    )

    route = data.get("route", {})

    # 解析路径
    paths = []
    for path in route.get("paths", []):
        parsed_path = _parse_path(path, "walking")
        paths.append(parsed_path)

    result = {
        "origin": route.get("origin", formatted_origin),
        "destination": route.get("destination", formatted_destination),
        "paths": paths,
    }

    _cache_set(cache_key, result, _DIRECTION_TTL_SECONDS)
    return result


def plan_bicycling_route(
    origin: str,
    destination: str,
) -> Dict:
    """骑行路径规划

    Args:
        origin: 起点经纬度，格式 "lng,lat" (经度在前，纬度在后)
        destination: 目的地经纬度，格式 "lng,lat"

    Returns:
        Dict: 路径规划结果，包含：
            - origin: 起点坐标
            - destination: 目的地坐标
            - paths: 路径列表，每条路径包含 distance(米) 和 duration(秒)

    Raises:
        AMapError: API 调用失败时
    """
    cache_key = f"bicycling:{origin}:{destination}"

    # 检查缓存
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    if not is_configured():
        raise AMapError("AMAP_API_KEY 未配置，无法使用路径规划功能")

    # 格式化经纬度
    formatted_origin = _format_location(origin)
    formatted_destination = _format_location(destination)

    # 请求高德 API
    data = _request(
        "/v5/direction/bicycling",
        params={
            "origin": formatted_origin,
            "destination": formatted_destination,
        },
        cache_ttl=_DIRECTION_TTL_SECONDS,
    )

    route = data.get("route", {})

    # 解析路径
    paths = []
    for path in route.get("paths", []):
        parsed_path = _parse_path(path, "bicycling")
        paths.append(parsed_path)

    result = {
        "origin": route.get("origin", formatted_origin),
        "destination": route.get("destination", formatted_destination),
        "paths": paths,
    }

    _cache_set(cache_key, result, _DIRECTION_TTL_SECONDS)
    return result


def plan_electrobike_route(
    origin: str,
    destination: str,
) -> Dict:
    """电动车路径规划

    Args:
        origin: 起点经纬度，格式 "lng,lat" (经度在前，纬度在后)
        destination: 目的地经纬度，格式 "lng,lat"

    Returns:
        Dict: 路径规划结果，包含：
            - origin: 起点坐标
            - destination: 目的地坐标
            - paths: 路径列表，每条路径包含 distance(米) 和 duration(秒)

    Raises:
        AMapError: API 调用失败时
    """
    cache_key = f"electrobike:{origin}:{destination}"

    # 检查缓存
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    if not is_configured():
        raise AMapError("AMAP_API_KEY 未配置，无法使用路径规划功能")

    # 格式化经纬度
    formatted_origin = _format_location(origin)
    formatted_destination = _format_location(destination)

    # 请求高德 API
    data = _request(
        "/v5/direction/electrobike",
        params={
            "origin": formatted_origin,
            "destination": formatted_destination,
        },
        cache_ttl=_DIRECTION_TTL_SECONDS,
    )

    route = data.get("route", {})

    # 解析路径
    paths = []
    for path in route.get("paths", []):
        parsed_path = _parse_path(path, "electrobike")
        paths.append(parsed_path)

    result = {
        "origin": route.get("origin", formatted_origin),
        "destination": route.get("destination", formatted_destination),
        "paths": paths,
    }

    _cache_set(cache_key, result, _DIRECTION_TTL_SECONDS)
    return result


def _parse_transit_segment(segment: Dict) -> Dict:
    """解析公交换乘路段

    Args:
        segment: 高德 API 返回的单个路段数据

    Returns:
        Dict: 解析后的路段数据
    """
    result = {}

    # 解析步行路段
    walking = segment.get("walking", {})
    if walking:
        result["walking"] = {
            "distance": int(walking.get("distance", 0)),
            "origin": walking.get("origin", ""),
            "destination": walking.get("destination", ""),
            "steps": [_parse_step(step) for step in walking.get("steps", [])],
        }

    # 解析公交路段
    bus = segment.get("bus", {})
    if bus and bus.get("buslines"):
        buslines = []
        for line in bus["buslines"]:
            busline = {
                "name": line.get("name", ""),
                "id": line.get("id", ""),
                "type": line.get("type", ""),
                "distance": int(line.get("distance", 0)),
                "departure_stop": {
                    "id": line.get("departure_stop", {}).get("id", ""),
                    "name": line.get("departure_stop", {}).get("name", ""),
                    "location": line.get("departure_stop", {}).get("location", ""),
                },
                "arrival_stop": {
                    "id": line.get("arrival_stop", {}).get("id", ""),
                    "name": line.get("arrival_stop", {}).get("name", ""),
                    "location": line.get("arrival_stop", {}).get("location", ""),
                },
                "via_num": int(line.get("via_num", 0)),
                "via_stops": [],
            }

            # 解析途经站点
            for stop in line.get("via_stops", []):
                busline["via_stops"].append({
                    "id": stop.get("id", ""),
                    "name": stop.get("name", ""),
                    "location": stop.get("location", ""),
                })

            buslines.append(busline)

        result["bus"] = {"buslines": buslines}

    # 解析地铁路段（结构与公交类似）
    railway = segment.get("railway", {})
    if railway and railway.get("railway_lines"):
        railway_lines = []
        for line in railway["railway_lines"]:
            railway_line = {
                "name": line.get("name", ""),
                "id": line.get("id", ""),
                "distance": int(line.get("distance", 0)),
                "departure_stop": {
                    "id": line.get("departure_stop", {}).get("id", ""),
                    "name": line.get("departure_stop", {}).get("name", ""),
                    "location": line.get("departure_stop", {}).get("location", ""),
                },
                "arrival_stop": {
                    "id": line.get("arrival_stop", {}).get("id", ""),
                    "name": line.get("arrival_stop", {}).get("name", ""),
                    "location": line.get("arrival_stop", {}).get("location", ""),
                },
                "via_num": int(line.get("via_num", 0)),
                "via_stops": [],
            }

            # 解析途经站点
            for stop in line.get("via_stops", []):
                railway_line["via_stops"].append({
                    "id": stop.get("id", ""),
                    "name": stop.get("name", ""),
                    "location": stop.get("location", ""),
                })

            railway_lines.append(railway_line)

        result["railway"] = {"railway_lines": railway_lines}

    return result


def plan_transit_route(
    origin: str,
    destination: str,
    city: str,
) -> Dict:
    """公交/地铁路径规划（含跨城）

    Args:
        origin: 起点经纬度，格式 "lng,lat" (经度在前，纬度在后)
        destination: 目的地经纬度，格式 "lng,lat"
        city: 城市名称（用于 citycode），同城和跨城规划

    Returns:
        Dict: 路径规划结果，包含：
            - origin: 起点坐标
            - destination: 目的地坐标
            - distance: 总距离（米）
            - transits: 换乘方案列表，每个方案包含：
                - distance: 总距离
                - walking_distance: 步行距离
                - nightflag: 是否夜间线路
                - segments: 路段列表（步行 + 公交/地铁）

    Raises:
        AMapError: API 调用失败时
    """
    cache_key = f"transit:{origin}:{destination}:{city}"

    # 检查缓存
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    if not is_configured():
        raise AMapError("AMAP_API_KEY 未配置，无法使用路径规划功能")

    # 格式化经纬度
    formatted_origin = _format_location(origin)
    formatted_destination = _format_location(destination)

    # 请求高德 API
    data = _request(
        "/v5/direction/transit/integrated",
        params={
            "origin": formatted_origin,
            "destination": formatted_destination,
            "city1": city,
            "city2": city,
        },
        cache_ttl=_DIRECTION_TTL_SECONDS,
    )

    route = data.get("route", {})

    # 解析换乘方案
    transits = []
    for transit in route.get("transits", []):
        parsed_transit = {
            "distance": int(transit.get("distance", 0)),
            "walking_distance": int(transit.get("walking_distance", 0)),
            "nightflag": transit.get("nightflag", "0"),
            "segments": [],
        }

        # 解析路段
        for segment in transit.get("segments", []):
            parsed_transit["segments"].append(_parse_transit_segment(segment))

        transits.append(parsed_transit)

    result = {
        "origin": route.get("origin", formatted_origin),
        "destination": route.get("destination", formatted_destination),
        "distance": int(route.get("distance", 0)),
        "transits": transits,
    }

    _cache_set(cache_key, result, _DIRECTION_TTL_SECONDS)
    return result


def plan_route(
    origin: str,
    destination: str,
    mode: str,
    city: Optional[str] = None,
) -> Dict:
    """通用路径规划接口

    根据交通方式自动选择对应的规划接口。

    Args:
        origin: 起点经纬度，格式 "lng,lat"
        destination: 目的地经纬度，格式 "lng,lat"
        mode: 交通方式，支持：
            - "driving": 驾车
            - "walking": 步行
            - "bicycling": 骑行
            - "electrobike": 电动车
            - "transit": 公交/地铁
        city: 城市名称（transit 模式必填）

    Returns:
        Dict: 路径规划结果

    Raises:
        AMapError: API 调用失败时
        ValueError: 不支持的交通方式
    """
    mode_map = {
        "driving": plan_driving_route,
        "walking": plan_walking_route,
        "bicycling": plan_bicycling_route,
        "electrobike": plan_electrobike_route,
        "transit": plan_transit_route,
    }

    planner = mode_map.get(mode)
    if not planner:
        raise ValueError(f"不支持的交通方式: {mode}，支持的选项: {list(mode_map.keys())}")

    if mode == "transit":
        if not city:
            raise ValueError("transit 模式需要提供 city 参数")
        return planner(origin, destination, city)

    return planner(origin, destination)
