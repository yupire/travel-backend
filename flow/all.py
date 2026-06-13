from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
import operator

class TravelState(TypedDict):
    messages: Annotated[list, operator.add]
    retry_count: int
    is_fallback: bool
    # 降级路径用到的中间数据
    pois: list[dict]
    weather: list[dict]
    clusters: dict
    daily_plans: list[dict]


# ── 路由函数：决定走哪条路 ──────────────────────────

def should_continue(state: TravelState) -> str:
    last = state["messages"][-1]

    # 1. LLM 正常输出了工具调用 → 继续执行工具
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # 2. LLM 没有工具调用，也没报错 → 正常完成
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        # 检查是不是真的完成了（有 generate_itinerary 的调用记录）
        if _is_complete(state):
            return END

    # 3. 重试次数超限 → 降级
    if state["retry_count"] >= 3:
        return "fallback"

    # 4. 其他情况（LLM 输出异常）→ 重试
    return "agent"


def _is_complete(state: TravelState) -> bool:
    """检查 generate_itinerary 是否已被调用"""
    for msg in state["messages"]:
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                if tc["name"] == "generate_itinerary":
                    return True
    return False


# ── Agent 节点：带重试计数 ──────────────────────────

def agent_node(state: TravelState) -> dict:
    try:
        response = llm.invoke(state["messages"])
        return {
            "messages": [response],
            "retry_count": state["retry_count"]  # 成功了不加
        }
    except Exception as e:
        # LLM 调用本身报错（超时、网络等）
        return {
            "messages": [],
            "retry_count": state["retry_count"] + 1
        }


# ── 降级节点：硬编码流水线 ─────────────────────────

def fallback_node(state: TravelState) -> dict:
    """直接调 API，不经过 LLM"""
    try:
        # 复用已有工具函数，但直接调用而不是通过 LLM
        pois = search_pois.func(city=state["city"])
        weather = get_weather.func(
            city=state["city"],
            dates=state["dates"]
        )
        clusters = cluster_pois.func(pois=pois, n_days=state["days"])
        classified = classify_indoor_outdoor.func(pois=pois)
        assigned = assign_clusters_to_days.func(
            clusters=clusters,
            weather=weather
        )

        daily_plans = []
        for date, cluster_id in assigned.items():
            day_pois = clusters[cluster_id]
            route = plan_route.func(date=date, pois=day_pois)
            daily_plans.append({
                "date": date,
                "pois": day_pois,
                "route": route,
                "weather": next(w for w in weather if w["date"] == date)
            })

        # 降级路径不调 LLM 生成说明，用模板代替
        result = _build_fallback_output(daily_plans)
        result["is_fallback"] = True

        return {
            "daily_plans": daily_plans,
            "is_fallback": True,
            "messages": [{"role": "assistant", "content": str(result)}]
        }

    except Exception as e:
        # 降级也失败了，返回错误信息
        return {
            "is_fallback": True,
            "messages": [{"role": "assistant",
                         "content": f"规划失败：{str(e)}"}]
        }


def _build_fallback_output(daily_plans: list) -> dict:
    """降级时用模板生成说明，不调 LLM"""
    days = []
    for plan in daily_plans:
        w = plan["weather"]
        reason = (
            f"{plan['date']} 天气{w['condition']}，"
            f"降水概率{w['rain_prob']}%，"
            f"安排{'室内为主' if w['rain_prob'] > 60 else '户外'}行程。"
        )
        days.append({
            "date": plan["date"],
            "weather": w["condition"],
            "reason": reason,   # 模板生成，非 LLM
            "spots": plan["pois"],
            "routes": plan["route"]
        })
    return {"days": days}


# ── 组装 Graph ────────────────────────────────────

graph = StateGraph(TravelState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_node("fallback", fallback_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "fallback": "fallback",
        "agent": "agent",   # 重试：直接回到 agent
        END: END
    }
)
graph.add_edge("tools", "agent")
graph.add_edge("fallback", END)

app = graph.compile()