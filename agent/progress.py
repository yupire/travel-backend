"""流式进度描述

把内部工具调用转成面向用户的中文进度文案，供 plan_trip_stream 在流式过程中
告诉用户「当前这一步在做什么」，而不暴露内部英文函数名。
"""
from __future__ import annotations


# 工具名 → 前端展示用的友好中文描述。让流式进度能告诉用户「这一步在做什么」，
# 而不暴露内部英文函数名。未登记的工具回退用工具名本身。
_TOOL_LABELS = {
    "list_supported_cities": "获取支持的城市列表",
    "get_tourist_spots": "查询热门景点",
    "geocode_spot_locations": "解析景点坐标",
    "get_city_weather": "查询天气",
    "get_food_recommendations": "推荐当地美食",
    "cluster_spots_geographically": "按地理位置聚类景点",
    "export_clusters_geojson": "导出景点聚类地图",
    "plan_driving_directions": "规划驾车路线",
    "plan_walking_directions": "规划步行路线",
    "plan_bicycling_directions": "规划骑行路线",
    "plan_electrobike_directions": "规划电动车路线",
    "plan_transit_directions": "规划公共交通路线",
    "plan_route_directions": "规划交通路线",
    "classify_spot_indoor_outdoor": "区分室内/室外景点",
}


def _describe_tool_call(tool_call: dict) -> str:
    """把一次工具调用转成一句面向用户的中文进度描述。

    在基础动作（如「查询天气」）后追加关键参数（城市 / 日期 / 出行方式），让进度更
    具体：例如「查询天气 · beijing 2026-06-20」。参数缺失时只返回动作本身。
    """
    name = tool_call.get("name") or "?"
    label = _TOOL_LABELS.get(name, name)
    args = tool_call.get("args") or {}

    # 只挑选少量对用户有意义的参数拼到描述后面，避免把整串参数堆给用户
    hints: list[str] = []
    for key in ("city", "date", "mode"):
        val = args.get(key)
        if val:
            hints.append(str(val))
    if not hints and isinstance(args.get("spot_names"), list) and args["spot_names"]:
        spots = args["spot_names"]
        preview = "、".join(str(s) for s in spots[:3])
        hints.append(preview + ("…" if len(spots) > 3 else ""))

    return f"{label} · {' '.join(hints)}" if hints else label
