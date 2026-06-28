"""LangGraph 旅行规划 Agent —— 路由决策与 Graph 组装

完整 Graph 流程：

                      ┌───────┐
                      │ START │
                      └───┬───┘
                          ▼
       tools 执行完   ┌─────────┐
       必回 agent     │  agent  │  调用绑定工具的 LLM 推理
    ┌────────────────►│  (LLM)  │
    │                 └────┬────┘
    │                      │ should_continue 按最后一条消息路由：
    │     ┌────────────────┼─────────────────┬──────────────────┐
    │     │ 有 tool_calls  │ 无 tool_calls    │ agent 自身异常    │ retry ≥ MAX_RETRY
    │     ▼                ▼                  │ 且 retry < MAX    ▼
    │  ┌───────┐      ┌──────────┐            │ ↑（回 agent     ┌──────────┐
    │  │ tools │      │  format  │            └──┘  重试）       │ fallback │
    │  │(ToolN)│      │ (JSON 模 │                              │(模板降级)│
    │  └───┬───┘      │  式 LLM) │                              └────┬─────┘
    │      │          └────┬─────┘                                   │
    └──────┘               │                                         │
   失败时汇总错误、         └──────────────────┬──────────────────────┘
   注入引导消息                                ▼
   （先修复再继续）                          ┌───────┐
                                            │  END  │
                                            └───┬───┘
                                                ▼
                            plan_trip 解析最终消息：
                            _extract_json_payload → _build_trip_response
                            （逐天校验 / 缺字段兜底）
                            → 始终返回合法完整的 TripResponse（绝不 500）

节点说明：
- agent   : 调用绑定工具的 LLM 推理，决定调用哪些工具 / 是否结束
- tools   : LangGraph ToolNode，按 LLM 决策执行 backend/tools 中已注册的工具；
            本包装层还会收集失败的工具调用并回灌引导消息，要求先修复再继续
- format  : LLM 不再调用工具时，用 JSON 模式 LLM 把对话里收集到的全部数据
            （工具返回的经纬度/门票/交通 + Agent 规划）整理成严格 TripResponse JSON
- fallback: 工具 / LLM 重试超限时的固定（确定性）降级路径，转入
            agent/fallback_graph 的子图重新规划（见该文件），产出合法 TripResponse

条件边（should_continue 的返回值 → path_map）：
- agent → tools    : LLM 在响应中产生了 tool_calls
- agent → agent    : agent 自身异常（[agent_node error]），且 retry < MAX_RETRY，重试
- agent → format   : LLM 不再调用工具，进入格式化节点产出严格 JSON
- agent → fallback : retry ≥ MAX_RETRY（异常/重试超限）
- tools → agent    : 工具执行完必回 agent，让 LLM 看结果再决定（固定边）
- format → END     : 固定边
- fallback → END   : 固定边
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agent.state import TravelState
from agent.nodes import agent_node, tools_node, format_node, fallback_node

logger = logging.getLogger(__name__)


# ──────────────────────── 条件边 ────────────────────────
MAX_RETRY = 3

# agent ↔ tools 主循环的软上限：一次完整规划里允许 LLM 发起工具调用的最大轮数。
# 每轮 agent 可一次性并行发起多个 tool_calls，正常一两轮就能把景点/天气/交通等
# 并行查完，8 轮足够覆盖「先查→看结果→补查→修正」的真实需求。达到上限后强制进入
# format，用「已经拿到的数据」收尾，而不是无限循环或撞上 recursion_limit 直接报错。
MAX_TOOL_ROUNDS = 8


def _count_tool_rounds(messages: list) -> int:
    """统计历史里 AI 发起过工具调用的轮数（含当前这条待执行的）。"""
    return sum(
        1 for m in messages
        if getattr(m, "type", "") == "ai" and (getattr(m, "tool_calls", None) or [])
    )


def should_continue(state: TravelState) -> str:
    """agent 节点后的路由决策。

    返回值映射到 add_conditional_edges 的 path_map：
    - "tools"        → 调用工具
    - "agent"        → agent 自身异常，未超限时重试一次
    - "fallback"     → 异常 / 重试超限，走降级
    - "format"       → LLM 不再调用工具 / 工具轮数达上限，进入格式化节点产出严格 JSON
    """
    # 1) 重试次数超限：直接降级
    retry = state.get("retry_count", 0)
    if retry >= MAX_RETRY:
        logger.warning("should_continue: 重试次数 %d 已达上限 %d -> fallback",
                       retry, MAX_RETRY)
        return "fallback"

    messages = state.get("messages") or []
    if not messages:
        logger.warning("should_continue: messages 为空 -> fallback")
        return "fallback"

    last = messages[-1]

    # 2) agent_node 自身异常（超时/限流）会塞入 [agent_node error] 的 SystemMessage，
    #    未超限时回到 agent 重试，让 LLM 再尝试一次。
    content = str(getattr(last, "content", "") or "")
    if getattr(last, "type", "") == "system" and content.startswith("[agent_node error]"):
        logger.warning("should_continue: agent 异常，retry=%d < %d -> 重试 agent",
                       retry, MAX_RETRY)
        return "agent"

    # 3) LLM 决定调用工具 → 走 tools 节点；但先卡主循环软上限：工具轮数达上限时
    #    不再执行新一轮工具，强制进入 format 用已收集数据收尾，避免无限循环/超慢。
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        rounds = _count_tool_rounds(messages)
        if rounds > MAX_TOOL_ROUNDS:
            logger.warning(
                "should_continue: 工具调用轮数 %d 超过上限 %d -> 强制 format 收尾",
                rounds, MAX_TOOL_ROUNDS)
            return "format"
        logger.info("should_continue: LLM 请求 %d 个工具（第 %d 轮）-> tools",
                    len(tool_calls), rounds)
        return "tools"

    # 4) LLM 不再调用工具：进入 format 节点，用 JSON 模式把已收集数据整理成严格
    #    TripResponse JSON（Agent 主链常输出 Markdown，不能直接当最终结果）。
    logger.info("should_continue: LLM 无工具调用 -> format")
    return "format"


# format 节点最多重试次数：每次 invoke 自带超时，超时即失败回到这里重试，
# 用尽后结束（_finalize 会用 Agent 原始输出兜底），保证不会卡死在 format。
MAX_FORMAT_RETRY = 2


def after_format(state: TravelState) -> str:
    """format 节点后的路由：成功就结束，失败（含超时）未超限则重跑 format。

    - 成功（format_done=True）             -> END
    - 失败且 format_retry < 上限           -> "format"（重试）
    - 失败且重试用尽                       -> END（由 _finalize 兜底，不再阻塞）
    """
    if state.get("format_done"):
        return END

    fmt_retry = state.get("format_retry", 0)
    if fmt_retry < MAX_FORMAT_RETRY:
        logger.warning("after_format: 格式化失败，重试 format (%d/%d)",
                       fmt_retry, MAX_FORMAT_RETRY)
        return "format"

    logger.error("after_format: 格式化重试 %d 次仍失败 -> 结束，交由 _finalize 兜底",
                 fmt_retry)
    return END


# ──────────────────────── 组装 Graph ────────────────────────
graph = StateGraph(TravelState)

# 节点
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.add_node("format", format_node)
graph.add_node("fallback", fallback_node)

# 入口
graph.add_edge(START, "agent")

# agent 后的条件边
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "agent": "agent",
        "format": "format",
        "fallback": "fallback",
        END: END,
    },
)

# 工具节点执行完一定回 agent（agent 看完结果再判断）
graph.add_edge("tools", "agent")

# format 后按结果路由：成功结束，失败（含超时）未超限则重跑 format
graph.add_conditional_edges(
    "format",
    after_format,
    {
        "format": "format",
        END: END,
    },
)

# fallback 完即结束
graph.add_edge("fallback", END)

app = graph.compile()
