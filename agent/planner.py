"""LangGraph 旅行规划 Agent

Graph 流程骨架：

    ┌─────┐   1. 需要工具   ┌────────┐
    │START├───────────────►│  tools │
    └──┬──┘                 └───┬────┘
       │                        │ 2. 工具结果返回
       │ 3. 推理                ▼
       │                    ┌────────┐
       └───────────────────►│ agent  │◄────────┐
                            └───┬────┘         │
                                │ 4. 无工具调用 │
                                ▼              │
                              ┌─────┐          │
                              │ END │  (回到第 1 步继续工具循环)
                              └─────┘

节点说明：
- agent  : 调用 LLM 推理，决定调用哪些工具 / 是否结束
- tools  : LangGraph ToolNode，按 LLM 决策执行 backend/tools 中已注册的工具
- fallback: 工具 / LLM 重试超限时的硬编码降级路径（不调 LLM，模板输出）

条件边：
- agent → tools   : LLM 在响应中产生了 tool_calls
- agent → fallback: LLM 异常 / 重试次数超限
- agent → END     : LLM 不再调用工具，视为完成
- tools → agent   : 工具执行完必回 agent，让 LLM 看结果再决定
"""
from __future__ import annotations

from typing import Annotated, TypedDict

import operator
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from models import TripRequest, TripResponse
from tools import ALL_TOOLS
from agent.prompts import PLAN_REASONING_PROMPT, build_initial_messages


# ──────────────────────── 大模型 ────────────────────────
# Agent 主链：（通过 OpenAI 兼容接口），带工具调用能力
_llm = ChatOpenAI(
    base_url="https://api.",
    api_key=os.getenv(""),
    model="",
    streaming=False,
    reasoning_effort="high",
    # 透传 OpenAI SDK 不直接支持的字段（如 thinking）
    model_kwargs={"extra_body": {"thinking": {"type": "enabled"}}},
).bind_tools(ALL_TOOLS)

# 用于最终生���自然语言 reasoning/summary 的 LLM（无工具绑定）
_reasoning_llm = ChatOpenAI(
    base_url="https://api.",
    api_key=os.getenv(""),
    model="",
    streaming=False,
    reasoning_effort="high",
    model_kwargs={"extra_body": {"thinking": {"type": "enabled"}}},
)


# ──────────────────────── State ────────────────────────
class TravelState(TypedDict):
    """贯穿整个 Graph 的状态。

    字段含义：
    - messages: LLM 的完整对话历史（Operator.add 累加）
    - city / dates / days: 用户输入的旅行参数
    - retry_count: agent 节点失败重试次数
    - is_fallback: 是否走了降级路径
    - pois / weather / clusters / daily_plans: 工具调用的中间数据
    """
    city: str
    dates: list[str]
    days: int
    messages: Annotated[list, operator.add]
    retry_count: int
    is_fallback: bool
    pois: list[dict]
    weather: list[dict]
    clusters: dict
    daily_plans: list[dict]


# ──────────────────────── 节点定义 ────────────────────────
def agent_node(state: TravelState) -> dict:
    """节点 2：调用 LLM 进行逻辑推理，决定调用工具或输出最终结果。"""
    # 如果 messages 为空，调用 prompts.py 中的工厂函数注入系统提示与用户问题
    messages = state.get("messages") or []
    if not messages:
        messages = build_initial_messages(
            city=state["city"],
            dates=state["dates"],
            days=state["days"],
        )

    try:
        response = _llm.invoke(messages)
        return {
            "messages": [response],
            "retry_count": state.get("retry_count", 0),
        }
    except Exception as e:
        # LLM 本身异常（超时/限流等），让路由函数走降级
        return {
            "messages": [
                SystemMessage(content=f"[agent_node error] {e!r}"),
            ],
            "retry_count": state.get("retry_count", 0) + 1,
        }


def reasoning_node(state: TravelState) -> dict:
    """节点 4：工具全部执行完，LLM 不再要工具时，调用 reasoning LLM 生成自然语言。

    之所以单独拆出来，是因为 agent_node 绑了工具后，模型有可能不再愿意纯文本输出；
    拆出一个无工具的 LLM 专门负责把数据渲染成中文 reasoning / summary。
    """
    chain = PLAN_REASONING_PROMPT | _reasoning_llm
    response = chain.invoke({
        "city": state["city"],
        "start_date": state["dates"][0],
        "end_date": state["dates"][-1],
        "days_json": str(state.get("daily_plans", [])),
    })
    return {"messages": [response]}


def fallback_node(state: TravelState) -> dict:
    """降级节点：LLM 多次失败时，硬编码用模板生成（不依赖 LLM）。"""
    daily_plans = state.get("daily_plans") or []
    days = []
    for i, plan in enumerate(daily_plans, start=1):
        w = plan.get("weather", {})
        days.append({
            "day": i,
            "date": plan.get("date"),
            "weather": w,
            "spots": plan.get("pois", []),
            "reasoning": (
                f"{plan.get('date')} 天气{w.get('condition', '未知')}，"
                "降级路径使用模板生成说明。"
            ),
            "is_indoor_outdoor_filter": w.get("condition") == "rainy",
        })
    result = {
        "city": state.get("city"),
        "start_date": state["dates"][0] if state.get("dates") else "",
        "end_date": state["dates"][-1] if state.get("dates") else "",
        "total_days": state.get("days", len(days)),
        "itinerary": days,
        "summary": "（降级输出：未经过 LLM 推理生成，仅使用工具数据 + 模板）",
    }
    return {
        "is_fallback": True,
        "messages": [SystemMessage(content=str(result))],
    }


# ──────────────────────── 条件边 ────────────────────────
MAX_RETRY = 3


def should_continue(state: TravelState) -> str:
    """agent 节点后的路由决策。

    返回值映射到 add_conditional_edges 的 path_map：
    - "tools"        → 调用工具
    - "reasoning"    → LLM 决定结束，进入 reasoning 节点
    - "fallback"     → 异常 / 重试超限，走降级
    - END            → 直接结束
    """
    # 1) 重试次数超限：直接降级
    if state.get("retry_count", 0) >= MAX_RETRY:
        return "fallback"

    messages = state.get("messages") or []
    if not messages:
        return "fallback"

    last = messages[-1]

    # 2) 最后一轮是工具返回（ToolMessage），让 LLM 接着看结果
    if getattr(last, "type", "") == "tool":
        return "agent"

    # 3) LLM 决定调用工具 → 走 tools 节点
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return "tools"

    # 4) LLM 没有 tool_calls，且没异常 → 进入 reasoning 节点生成自然语言
    return "reasoning"


# ──────────────────────── 组装 Graph ────────────────────────
graph = StateGraph(TravelState)

# 节点
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(ALL_TOOLS))
graph.add_node("reasoning", reasoning_node)
graph.add_node("fallback", fallback_node)

# 入口
graph.add_edge(START, "agent")

# agent 后的条件边
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "reasoning": "reasoning",
        "fallback": "fallback",
        END: END,
    },
)

# 工具节点执行完一定回 agent（agent 看完结果再判断）
graph.add_edge("tools", "agent")

# reasoning 与 fallback 完即结束
graph.add_edge("reasoning", END)
graph.add_edge("fallback", END)

app = graph.compile()


# ──────────────────────── 对外入口 ────────────────────────
def plan_trip(request: TripRequest) -> TripResponse:
    """调用 Agent，串行触发整个 Graph，返回结构化 TripResponse。"""
    from datetime import datetime, timedelta

    start = datetime.fromisoformat(request.start_date)
    end = datetime.fromisoformat(request.end_date)
    dates = [
        (start + timedelta(days=i)).date().isoformat()
        for i in range((end - start).days + 1)
    ]

    final_state = app.invoke({
        "city": request.city,
        "dates": dates,
        "days": len(dates),
        "messages": [],
        "retry_count": 0,
        "is_fallback": False,
        "pois": [],
        "weather": [],
        "clusters": {},
        "daily_plans": [],
    })

    # 从最后一条 assistant 消息中提取 JSON 字符串
    import json
    last = final_state["messages"][-1]
    content = getattr(last, "content", "") or ""

    if isinstance(content, list):
        # Anthropic 部分版本 content 是 list[dict]
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        payload = {
            "city": request.city,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "total_days": len(dates),
            "itinerary": [],
            "summary": content or "（无输出）",
        }

    return TripResponse(**payload)
