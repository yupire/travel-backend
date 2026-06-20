"""固定（确定性）行程规划子图 —— Agent 主链失败时的降级路径。

与 planner.py 的 Agent 主链不同：本图是**硬编码、固定边**的流水线，不依赖
LLM 自主决策；它直接调用 backend/tools 里的底层函数拿数据，按固定拓扑产出
一份合法的 TripResponse。只有两处「锦上添花」的步骤会调用 LLM（按天气分配、
生成理由），且两者都允许失败降级，失败也不影响主流程出结果。

固定图拓扑（节点编号对应需求）：

                         ┌───────┐
                         │ START │
                         └───┬───┘
              ┌──────────────┴──────────────┐
              ▼ 并行                          ▼ 并行
        ┌───────────┐                  ┌───────────┐
        │ 节点2      │                  │ 节点3      │
        │ 查询景点   │                  │ 查询天气   │
        │ +地理聚类  │                  └─────┬─────┘
        │ +室内外    │                        │
        └─────┬─────┘                        │
              └──────────────┬───────────────┘
                             ▼ （两路汇合）
                       ┌───────────┐
                       │ 节点4      │ 调 LLM：按天气重排天组
                       │ 天气分配   │ （失败→跳过，保留地理顺序）
                       └─────┬─────┘
                             ▼
                       ┌───────────┐
                       │ 节点5      │ 每日路线规划
                       │ 路线规划   │ （距离最优 + 交通估算）
                       └─────┬─────┘
                             ▼
                       ┌───────────┐
                       │ 节点6      │ 调 LLM：生成每日理由 + 总结
                       │ 理由生成   │ （失败→无理由，模板总结）
                       └─────┬─────┘
                             ▼
                       ┌───────────┐
                       │ 节点7      │ 组装 TripResponse JSON
                       │ 结构化输出 │
                       └─────┬─────┘
                             ▼
                         ┌───────┐
                         │  END  │
                         └───────┘
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Annotated, TypedDict

import operator

from langchain_openai import ChatOpenAI

from models import TripResponse
from agent.prompts import PLAN_REASONING_PROMPT

from tools.spots import get_top_spots
from tools.clusters import cluster_spots_by_geo
from tools.spot_indoor import classify_spot_indoor
from tools.weather import get_weather
from tools.routing import optimize_by_distance_progression, add_transport_info

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)


# ──────────────────────── LLM 客户端 ────────────────────────
# 与 planner.py 同样的 DeepSeek 配置；这里独立构造以避免与 planner 形成循环导入。
# 节点4 用 JSON 模式（要结构化排列结果），节点6 用普通文本链（生成中文理由）。
_assign_llm = ChatOpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-v4-pro",
    streaming=False,
    model_kwargs={"response_format": {"type": "json_object"}},
)

_reason_llm = ChatOpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-v4-pro",
    streaming=False,
)


# ──────────────────────── State ────────────────────────
class FallbackState(TypedDict, total=False):
    """贯穿固定子图的状态。

    节点2 / 节点3 并行写入不同字段（spots/clusters vs weather），互不冲突；
    其余节点串行推进，逐步把 daily_spots 收敛成可直接序列化的 itinerary。
    """
    # —— 输入 ——
    city: str
    dates: list[str]
    days: int
    # —— 节点2 产出 ——
    spots: list[dict]          # 全部候选景点（含 lat/lng/type/is_indoor）
    clusters: list[dict]       # 地理聚类结果
    # —— 节点3 产出 ——
    weather: dict              # {date: weatherinfo|None}
    # —— 节点4/5 产出 ——
    daily_spots: list[list[dict]]   # 逐日景点（已分配、已排线）
    used_weather_filter: bool       # 是否成功按天气重排
    # —— 节点6 产出 ——
    reasonings: dict           # {day(int): reasoning(str)}
    summary: str
    # —— 节点7 产出 ——
    itinerary: list[dict]      # 逐日 DayPlan 形状的最终行程
    # —— 步骤计数（仅日志用，operator.add 兼容并行写入）——
    step: Annotated[int, operator.add]


# ──────────────────────── 默认值 ────────────────────────
_DEFAULT_WEATHER = {
    "condition": "cloudy",
    "temp_high": 0,
    "temp_low": 0,
    "description": "暂无天气数据",
}


# ──────────────────────── 节点2：查询景点 + 聚类 + 室内外 ────────────────────────
def query_spots_node(state: FallbackState) -> dict:
    """节点2：拉取候选景点 → 室内外分类 → 地理聚类（初步行程的地理骨架）。"""
    city = state["city"]
    days = state.get("days", 1)
    # 候选数量按天数放大，给后续分配留余量（至少 8 个）
    limit = max(days * 4, 8)
    spots = get_top_spots(city, limit=limit)
    logger.info("【fallback 节点2】查询景点 city=%s limit=%d → %d 个", city, limit, len(spots))

    # 室内外分类：补齐/规整每个景点的 is_indoor 与 type 字段
    for s in spots:
        is_indoor = classify_spot_indoor(
            is_indoor=s.get("is_indoor"),
            spot_type=s.get("type"),
            name=s.get("name", ""),
            tags=s.get("tags", []),
        )
        s["is_indoor"] = is_indoor
        s["type"] = "indoor" if is_indoor else "outdoor"

    # 地理聚类（DBSCAN）：把景点分成若干地理区域，作为分天的骨架
    clusters = cluster_spots_by_geo(spots) if spots else []
    logger.info("【fallback 节点2】地理聚类 → %d 个簇", len(clusters))
    return {"spots": spots, "clusters": clusters, "step": 1}


# ──────────────────────── 节点3：查询天气（与节点2并行）────────────────────────
def query_weather_node(state: FallbackState) -> dict:
    """节点3：逐日查询天气；任一天失败记 None，不阻断流程。"""
    city = state["city"]
    weather: dict = {}
    for d in state.get("dates", []):
        w = get_weather(city, d)  # 失败返回 None
        weather[d] = w
    ok = sum(1 for v in weather.values() if v)
    logger.info("【fallback 节点3】查询天气 city=%s → %d/%d 天命中", city, ok, len(weather))
    return {"weather": weather, "step": 1}


# ──────────────────────── 分天骨架（确定性）────────────────────────
def _baseline_day_groups(
    clusters: list[dict],
    spots: list[dict],
    days: int,
) -> list[list[dict]]:
    """把景点确定性地分到 days 个天组。

    思路：以地理聚类为「不可拆分单元」，按簇顺序铺平成一条地理相邻的景点序列，
    再均匀切成 days 段。这样既保证同一天的景点地理接近，又不会某天空着。
    """
    smap = {s["id"]: s for s in spots}
    used: set[str] = set()
    flat: list[dict] = []
    # 先按簇（已按 size 降序）顺序铺平
    for c in clusters or []:
        for sid in c.get("spot_ids", []):
            s = smap.get(sid)
            if s and sid not in used:
                flat.append(s)
                used.add(sid)
    # 未进入任何簇的散点接在后面
    for s in spots:
        if s["id"] not in used:
            flat.append(s)
            used.add(s["id"])

    groups: list[list[dict]] = [[] for _ in range(max(days, 1))]
    if not flat:
        return groups
    per = max(1, math.ceil(len(flat) / max(days, 1)))
    for i, s in enumerate(flat):
        groups[min(i // per, days - 1)].append(s)
    return groups


# ──────────────────────── 节点4：按天气分配（LLM 重排天组）────────────────────────
def _llm_weather_order(
    groups: list[list[dict]],
    dates: list[str],
    weather: dict,
) -> list[int]:
    """让 LLM 决定把哪个天组排到哪一天，使雨天对应室内景点多的天组。

    返回一个 groups 索引的排列 order：第 i 天使用 groups[order[i]]。
    任何异常（调用失败 / 输出非法 / 非排列）都向上抛，由调用方降级。
    """
    group_brief = [
        {
            "group": i,
            "indoor": sum(1 for s in g if s.get("is_indoor")),
            "outdoor": sum(1 for s in g if not s.get("is_indoor")),
            "spots": [s.get("name", "") for s in g],
        }
        for i, g in enumerate(groups)
    ]
    day_brief = [
        {"day": i + 1, "date": d,
         "condition": (weather.get(d) or {}).get("condition", "unknown")}
        for i, d in enumerate(dates)
    ]
    prompt = (
        "你在做旅行行程的天气适配。下面有若干『天组』（每组是地理相邻的一批景点，"
        "含室内/室外数量）和每一天的天气。请把天组分配到各天，规则：\n"
        "- 雨天（rainy）优先安排室内景点多的天组；\n"
        "- 晴/多云天优先安排室外景点多的天组；\n"
        "- 每个天组恰好用一次，是一一对应的排列。\n\n"
        f"天组：{json.dumps(group_brief, ensure_ascii=False)}\n"
        f"每天天气：{json.dumps(day_brief, ensure_ascii=False)}\n\n"
        '只输出 JSON，格式：{"order": [<第1天用的group编号>, <第2天>, ...]}，'
        "order 必须是 0..N-1 的一个排列（N=天组数）。"
    )
    resp = _assign_llm.invoke(prompt)
    obj = json.loads(str(resp.content))
    order = obj.get("order")
    if (
        not isinstance(order, list)
        or sorted(order) != list(range(len(groups)))
    ):
        raise ValueError(f"LLM 返回的 order 非法：{order!r}")
    return [int(i) for i in order]


def assign_by_weather_node(state: FallbackState) -> dict:
    """节点4：先确定性分天组，再调 LLM 按天气重排；LLM 失败则保留地理顺序。"""
    spots = state.get("spots", [])
    clusters = state.get("clusters", [])
    dates = state.get("dates", [])
    days = state.get("days", len(dates) or 1)
    weather = state.get("weather", {})

    groups = _baseline_day_groups(clusters, spots, days)
    used_filter = False
    try:
        order = _llm_weather_order(groups, dates, weather)
        groups = [groups[i] for i in order]
        used_filter = True
        logger.info("【fallback 节点4】LLM 按天气重排天组成功：order=%s", order)
    except Exception as e:
        logger.warning("【fallback 节点4】LLM 天气分配失败，跳过按天气分配：%r", e)

    return {"daily_spots": groups, "used_weather_filter": used_filter, "step": 1}


# ──────────────────────── 节点5：每日路线规划 ────────────────────────
def _anchor(state: FallbackState) -> tuple[float, float]:
    """路线起锚点：优先用最大簇中心，否则全部景点质心。"""
    clusters = state.get("clusters") or []
    if clusters:
        c = clusters[0].get("center") or {}
        if "lat" in c and "lng" in c:
            return (c["lat"], c["lng"])
    spots = state.get("spots") or []
    if spots:
        n = len(spots)
        return (sum(s["lat"] for s in spots) / n, sum(s["lng"] for s in spots) / n)
    return (0.0, 0.0)


def route_plan_node(state: FallbackState) -> dict:
    """节点5：按距离最优重排每天景点顺序，并补充相邻景点的交通估算。"""
    daily_spots = state.get("daily_spots", [])
    anchor = _anchor(state)
    # 跨天「由近及远」排序，避免来回折返
    ordered = optimize_by_distance_progression(daily_spots, anchor)
    # 每天内补充 transport_from_prev
    ordered = [add_transport_info(day) for day in ordered]
    logger.info("【fallback 节点5】路线规划完成：共 %d 天", len(ordered))
    return {"daily_spots": ordered, "step": 1}


# ──────────────────────── 节点6：生成理由（LLM，可失败）────────────────────────
def reasoning_node(state: FallbackState) -> dict:
    """节点6：调 LLM 为每日生成理由 + 整体总结；失败则不给理由、用模板总结。"""
    dates = state.get("dates", [])
    days = state.get("days", len(dates) or 1)
    daily_spots = state.get("daily_spots", [])
    weather = state.get("weather", {})
    city = state.get("city", "")

    reasonings: dict[int, str] = {}
    summary = "（降级路径输出：基于真实景点 / 天气数据按固定流程生成）"

    days_payload = [
        {
            "day": i + 1,
            "date": dates[i] if i < len(dates) else "",
            "weather": (weather.get(dates[i]) if i < len(dates) else None) or _DEFAULT_WEATHER,
            "spots": [s.get("name", "") for s in (daily_spots[i] if i < len(daily_spots) else [])],
        }
        for i in range(days)
    ]

    try:
        messages = PLAN_REASONING_PROMPT.format_messages(
            city=city,
            start_date=dates[0] if dates else "",
            end_date=dates[-1] if dates else "",
            days_json=json.dumps(days_payload, ensure_ascii=False),
        )
        resp = _reason_llm.invoke(messages)
        obj = json.loads(str(resp.content))
        for d in obj.get("days", []):
            day_no = int(d.get("day", 0))
            if day_no:
                reasonings[day_no] = str(d.get("reasoning", ""))
        if obj.get("summary"):
            summary = str(obj["summary"])
        logger.info("【fallback 节点6】LLM 生成理由成功：%d 天有理由", len(reasonings))
    except Exception as e:
        logger.warning("【fallback 节点6】LLM 生成理由失败，行程不附理由：%r", e)

    return {"reasonings": reasonings, "summary": summary, "step": 1}


# ──────────────────────── 节点7：结构化输出 ────────────────────────
def _to_spot_plan(s: dict) -> dict:
    """把内部景点 dict 收敛成 SpotPlan 形状（缺字段补合理默认）。"""
    is_indoor = bool(s.get("is_indoor", s.get("type") == "indoor"))
    return {
        "id": s.get("id", ""),
        "name": s.get("name", ""),
        "lat": s.get("lat", 0.0),
        "lng": s.get("lng", 0.0),
        "duration_min": s.get("duration_min", 90),
        "open_time": s.get("open_time", "09:00-18:00"),
        "ticket": s.get("ticket", 0),
        "type": "indoor" if is_indoor else "outdoor",
        "is_indoor": is_indoor,
        "tags": s.get("tags", []),
        "description": s.get("description", ""),
        "nearby_foods": [],
        "transport_from_prev": s.get("transport_from_prev"),
    }


def output_node(state: FallbackState) -> dict:
    """节点7：组装逐日 itinerary，落到 state.itinerary（由 run_fallback 收尾成 TripResponse）。"""
    dates = state.get("dates", [])
    days = state.get("days", len(dates) or 1)
    daily_spots = state.get("daily_spots", [])
    weather = state.get("weather", {})
    reasonings = state.get("reasonings", {})

    itinerary: list[dict] = []
    for i in range(days):
        date = dates[i] if i < len(dates) else ""
        w = (weather.get(date) if date else None) or _DEFAULT_WEATHER
        winfo = {
            "condition": w.get("condition", "cloudy"),
            "temp_high": int(w.get("temp_high", 0) or 0),
            "temp_low": int(w.get("temp_low", 0) or 0),
            "description": w.get("description", "暂无天气数据"),
        }
        spots = [_to_spot_plan(s) for s in (daily_spots[i] if i < len(daily_spots) else [])]
        itinerary.append({
            "day": i + 1,
            "date": date,
            "weather": winfo,
            "spots": spots,
            "reasoning": reasonings.get(i + 1, ""),
            "is_indoor_outdoor_filter": winfo["condition"] == "rainy",
        })

    logger.info("【fallback 节点7】结构化输出完成：共 %d 天", len(itinerary))
    return {"itinerary": itinerary, "step": 1}


# ──────────────────────── 组装固定图 ────────────────────────
_graph = StateGraph(FallbackState)
_graph.add_node("spots", query_spots_node)       # 节点2
_graph.add_node("weather", query_weather_node)   # 节点3
_graph.add_node("assign", assign_by_weather_node)  # 节点4
_graph.add_node("route", route_plan_node)        # 节点5
_graph.add_node("reasoning", reasoning_node)     # 节点6
_graph.add_node("output", output_node)           # 节点7

# 节点2 / 节点3 从 START 并行扇出
_graph.add_edge(START, "spots")
_graph.add_edge(START, "weather")
# 两路汇合到节点4（assign 等两条入边都完成才执行）
_graph.add_edge("spots", "assign")
_graph.add_edge("weather", "assign")
# 之后固定串行：分配 → 路线 → 理由 → 输出 → END
_graph.add_edge("assign", "route")
_graph.add_edge("route", "reasoning")
_graph.add_edge("reasoning", "output")
_graph.add_edge("output", END)

fallback_app = _graph.compile()


# ──────────────────────── 对外入口 ────────────────────────
def run_fallback(city: str, dates: list[str], days: int) -> TripResponse:
    """运行固定子图，返回一份合法的 TripResponse（绝不抛异常）。"""
    logger.warning("进入固定降级子图：city=%s, 天数=%d", city, days)
    init: FallbackState = {
        "city": city,
        "dates": dates,
        "days": days,
        "step": 0,
    }
    try:
        final = fallback_app.invoke(init)
    except Exception as e:
        # 子图本身意外崩溃也要兜底，返回结构完整的空行程
        logger.exception("固定降级子图执行异常，返回最小安全结构：%r", e)
        return TripResponse(
            city=city,
            start_date=dates[0] if dates else "",
            end_date=dates[-1] if dates else "",
            total_days=len(dates),
            itinerary=[],
            summary="（降级子图执行失败，未能生成行程）",
        )

    return TripResponse(
        city=city,
        start_date=dates[0] if dates else "",
        end_date=dates[-1] if dates else "",
        total_days=days,
        itinerary=final.get("itinerary", []),
        summary=final.get("summary", "（降级路径输出）"),
    )
