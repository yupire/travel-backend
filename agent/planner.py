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

import logging
import operator
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils import convert_to_secret_str
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from models import DayPlan, TripRequest, TripResponse
from tools import ALL_TOOLS
from agent.prompts import build_initial_messages


# ──────────────────────── 日志 ────────────────────────
# 模块级 logger，命名为 "agent.planner"，便于在外层统一配置 handler / level。
# 这里只取 logger，不调用 basicConfig —— 由应用入口决定日志输出格式与级别，
# 避免库代码抢占根 logger 配置。
logger = logging.getLogger(__name__)


# ──────────────────────── 大模型 ────────────────────────
# Agent 主链：调用 （OpenAI 兼容协议），带工具调用能力
# 对应 OpenAI SDK 用法（参见 https://api-docs.deepseek.com）：
#   client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
#   client.chat.completions.create(model="deepseek-v4-pro", ...)
_llm = ChatOpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-v4-pro",
    streaming=False,
    reasoning_effort="high",
    # 透传 OpenAI SDK 不直接支持的字段（如 thinking）
    model_kwargs={"extra_body": {"thinking": {"type": "enabled"}}},
).bind_tools(ALL_TOOLS)

# ──────────────────────── State ────────────────────────
class TravelState(TypedDict):
    """贯穿整个 Graph 的状态。

    字段含义：
    - messages: LLM 的完整对话历史（Operator.add 累加）
    - city / dates / days: 用户输入的旅行参数
    - retry_count: agent 节点失败重试次数
    - is_fallback: 是否走了降级路径
    - step: 全局步骤计数器，每进入一个节点 +1，用于日志里标注「第几步」
    - pois / weather / clusters / daily_plans: 工具调用的中间数据
    - tool_errors: 最近一次 tools_node 执行中失败的工具调用（供路由判定是否需先修复）
    """
    city: str
    dates: list[str]
    days: int
    messages: Annotated[list, operator.add]
    retry_count: int
    is_fallback: bool
    step: int
    pois: list[dict]
    weather: list[dict]
    clusters: dict
    daily_plans: list[dict]
    tool_errors: list[dict]


# ──────────────────────── 节点定义 ────────────────────────
def agent_node(state: TravelState) -> dict:
    """节点 2：调用 LLM 进行逻辑推理，决定调用工具或输出最终结果。"""
    step = state.get("step", 0) + 1
    retry = state.get("retry_count", 0)
    logger.info("【第 %d 步】进入 agent_node[LLM 推理]：city=%s, 历史消息数=%d, 当前重试=%d",
                step, state.get("city"), len(state.get("messages") or []), retry)

    # 如果 messages 为空，调用 prompts.py 中的工厂函数注入系统提示与用户问题
    messages = state.get("messages") or []
    if not messages:
        logger.debug("【第 %d 步】messages 为空，构造初始系统提示 + 用户问题", step)
        messages = build_initial_messages(
            city=state["city"],
            dates=state["dates"],
            days=state["days"],
        )

    try:
        # 调用绑定了工具的主链：LLM 自行决定是直接回答还是产生 tool_calls
        response = _llm.invoke(messages)
        # 记录本轮 LLM 是否要求调用工具，便于排查 Graph 走向
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            logger.info("【第 %d 步】agent_node LLM 决定调用 %d 个工具：%s",
                        step, len(tool_calls),
                        [tc.get("name") for tc in tool_calls])
        else:
            logger.info("【第 %d 步】agent_node LLM 返回：无工具调用（准备结束/进入 reasoning）",
                        step)
        return {
            "messages": [response],
            "retry_count": retry,
            "step": step,
        }
    except Exception as e:
        # LLM 本身异常（超时/限流等），让路由函数走降级
        logger.exception("【第 %d 步】agent_node LLM 调用异常，retry_count -> %d：%r",
                         step, retry + 1, e)
        return {
            "messages": [
                SystemMessage(content=f"[agent_node error] {e!r}"),
            ],
            "retry_count": retry + 1,
            "step": step,
        }


# 预构建的 ToolNode：真正按 LLM 决策执行 backend/tools 里注册的工具。
# 我们用下面的 tools_node 包一层，只为在执行前后打印日志，执行委托给它。
_tool_node = ToolNode(ALL_TOOLS)


def _is_failed_tool_message(message, content: str) -> bool:
    """判断一条 ToolMessage 是否代表工具调用失败。

    覆盖两类失败：
    1. 工具抛异常被 ToolNode 捕获 —— status == "error"；
    2. 工具正常返回但内容为空/None（如天气接口失败返回 None，会序列化成
       "null"/"None"/""），这类「软失败」同样应让 LLM 重试修复。
    """
    if getattr(message, "status", None) == "error":
        return True
    normalized = content.strip().lower()
    if normalized in ("", "null", "none", "[]", "{}"):
        return True
    # 工具内部约定的错误结构，如 {"error": "..."}
    if normalized.startswith("{") and '"error"' in normalized:
        return True
    return False


def tools_node(state: TravelState) -> dict:
    """节点 3：执行 LLM 请求的工具，并逐个打印「第几步调用了哪个工具函数」。

    prebuilt ToolNode 本身是黑盒，不会告诉外层到底跑了哪些工具；这里包一层，
    在委托给 ToolNode 前后分别记录请求的工具名/参数与每个工具的返回。
    """
    step = state.get("step", 0) + 1
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    tool_calls = getattr(last, "tool_calls", None) or []

    logger.info("【第 %d 步】进入 tools_node[执行工具]：共 %d 个工具待执行",
                step, len(tool_calls))
    for i, tc in enumerate(tool_calls, start=1):
        logger.info("【第 %d 步】  ├─ 调用工具函数 #%d：%s，参数=%s",
                    step, i, tc.get("name"), tc.get("args"))

    # 真正执行交给 prebuilt ToolNode
    result = _tool_node.invoke(state)

    # ToolNode 返回 {"messages": [ToolMessage, ...]}，逐条记录工具产出
    new_msgs = result.get("messages", []) if isinstance(result, dict) else []
    tool_errors: list[dict] = []
    for m in new_msgs:
        content = str(getattr(m, "content", ""))
        preview = content if len(content) <= 200 else content[:200] + "…"
        name = getattr(m, "name", "?")
        logger.info("【第 %d 步】  └─ 工具 %s 返回：内容长度=%d，预览=%s",
                    step, name, len(content), preview)
        if _is_failed_tool_message(m, content):
            tool_errors.append({"tool": name, "detail": content[:300]})
            logger.warning("【第 %d 步】  ⚠ 工具 %s 调用失败：%s", step, name, content[:200])

    out = dict(result) if isinstance(result, dict) else {"messages": new_msgs}
    out["step"] = step
    out["tool_errors"] = tool_errors

    # 把失败的工具调用「收集起来」，并以引导消息要求 LLM 先修复再继续，
    # 避免在缺少关键数据（如天气）时直接进入最终行程输出。
    if tool_errors:
        names = "、".join(sorted({e["tool"] for e in tool_errors}))
        detail_lines = "\n".join(f"- {e['tool']}: {e['detail']}" for e in tool_errors)
        out["messages"] = list(new_msgs) + [
            HumanMessage(content=(
                f"⚠ 上一步有 {len(tool_errors)} 个工具调用失败（{names}）：\n"
                f"{detail_lines}\n\n"
                "请先修复这些失败：检查并修正参数后重试，例如天气查询失败时换用城市名"
                "或确认日期在预报窗口内。在成功拿到这些关键数据之前，不要直接输出最终行程 JSON。"
            ))
        ]
    return out


def fallback_node(state: TravelState) -> dict:
    """降级节点：LLM 多次失败时，硬编码用模板生成（不依赖 LLM）。"""
    step = state.get("step", 0) + 1
    daily_plans = state.get("daily_plans") or []
    logger.warning("【第 %d 步】进入 fallback_node[降级路径]：city=%s, daily_plans 天数=%d",
                   step, state.get("city"), len(daily_plans))
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
    logger.warning("【第 %d 步】fallback_node 模板生成完成：共 %d 天行程", step, len(days))
    return {
        "is_fallback": True,
        "messages": [SystemMessage(content=str(result))],
        "step": step,
    }


# ──────────────────────── 条件边 ────────────────────────
MAX_RETRY = 3


def should_continue(state: TravelState) -> str:
    """agent 节点后的路由决策。

    返回值映射到 add_conditional_edges 的 path_map：
    - "tools"        → 调用工具
    - "agent"        → agent 自身异常，未超限时重试一次
    - "fallback"     → 异常 / 重试超限，走降级
    - END            → LLM 不再调用工具，已输出最终 TripResponse JSON，直接结束
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

    # 3) LLM 决定调用工具 → 走 tools 节点
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        logger.info("should_continue: LLM 请求 %d 个工具 -> tools", len(tool_calls))
        return "tools"

    # 4) LLM 不再调用工具：此时它应已按 SYSTEM_PROMPT 输出完整 TripResponse JSON，
    #    直接结束，由 plan_trip 解析。不再经过 reasoning 节点（它输出的是
    #    {days, summary} 残缺 schema，会覆盖掉完整 JSON 导致 TripResponse 校验失败）。
    logger.info("should_continue: LLM 无工具调用，视为已产出最终 JSON -> END")
    return END


# ──────────────────────── 组装 Graph ────────────────────────
graph = StateGraph(TravelState)

# 节点
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
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
        "fallback": "fallback",
        END: END,
    },
)

# 工具节点执行完一定回 agent（agent 看完结果再判断）
graph.add_edge("tools", "agent")

# fallback 完即结束
graph.add_edge("fallback", END)

app = graph.compile()


# ──────────────────────── 输出解析 ────────────────────────
def _extract_json_payload(content) -> dict | None:
    """从 LLM 最终消息中尽力提取 JSON 对象。

    容忍三种常见脏输出：
    1. content 是 list[dict]（部分模型分块返回）；
    2. 用 ```json ... ``` 代码块包裹；
    3. JSON 前后夹带解释性文字。
    解析失败返回 None。
    """
    import json

    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    text = str(content or "").strip()
    if not text:
        return None

    # 去掉 ```json ... ``` 代码块包裹
    if text.startswith("```"):
        text = text.strip("`").strip()
        nl = text.find("\n")
        if nl != -1 and text[:nl].strip().lower() in ("json", ""):
            text = text[nl + 1:].strip()

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (TypeError, ValueError):
        pass

    # 截取第一个 { 到最后一个 }，容忍前后多余文字
    l, r = text.find("{"), text.rfind("}")
    if l != -1 and r > l:
        try:
            obj = json.loads(text[l:r + 1])
            return obj if isinstance(obj, dict) else None
        except (TypeError, ValueError):
            return None
    return None


def _build_trip_response(
    payload: dict | None,
    request: TripRequest,
    dates: list[str],
    raw_text: str,
) -> TripResponse:
    """把任意解析结果强制收敛成一个合法、完整的 TripResponse，绝不抛异常。

    - 顶层必填标量缺失 → 用请求参数补齐；
    - itinerary 逐天校验，脏数据的单天会被跳过而非拖垮整体；
    - summary 缺失 → 用 reasoning 文本 / 原始输出兜底。
    这样即便 Agent 没能给出规范行程，/plan 也总能返回结构完整的 JSON（而非 500）。
    """
    from pydantic import ValidationError

    payload = dict(payload) if isinstance(payload, dict) else {}

    payload.setdefault("city", request.city)
    payload.setdefault("start_date", request.start_date)
    payload.setdefault("end_date", request.end_date)
    payload.setdefault("total_days", len(dates))

    # 逐天校验 itinerary，丢弃无法解析的天
    raw_itinerary = payload.get("itinerary")
    valid_days: list[dict] = []
    if isinstance(raw_itinerary, list):
        for i, day in enumerate(raw_itinerary, start=1):
            try:
                valid_days.append(DayPlan(**day).model_dump())
            except (ValidationError, TypeError) as e:
                logger.warning("itinerary 第 %d 天校验失败，已跳过：%s", i, e)
    payload["itinerary"] = valid_days

    # summary 兜底：兼容 reasoning 残缺 schema 里的 days 文本
    if not payload.get("summary"):
        days_text = str(payload["days"]) if payload.get("days") else ""
        payload["summary"] = days_text or raw_text or "（未能生成行程说明）"

    try:
        resp = TripResponse(**payload)
        if not valid_days:
            logger.warning("plan_trip 输出无有效行程天，仅返回 summary 兜底结构")
        return resp
    except ValidationError as e:
        logger.error("TripResponse 最终校验仍失败，返回最小安全结构：%s", e)
        return TripResponse(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            total_days=len(dates),
            itinerary=[],
            summary=payload.get("summary") or raw_text or "（无输出）",
        )


# ──────────────────────── 对外入口 ────────────────────────
def plan_trip(request: TripRequest) -> TripResponse:
    """调用 Agent，串行触发整个 Graph，返回结构化 TripResponse。"""
    from datetime import datetime, timedelta

    logger.info("plan_trip 开始：city=%s, %s ~ %s",
                request.city, request.start_date, request.end_date)

    # 把起止日期展开成逐日列表（含首尾），供各节点按天处理
    start = datetime.fromisoformat(request.start_date)
    end = datetime.fromisoformat(request.end_date)
    dates = [
        (start + timedelta(days=i)).date().isoformat()
        for i in range((end - start).days + 1)
    ]
    logger.info("plan_trip 行程共 %d 天", len(dates))

    # 同步触发整个 Graph：从 START -> agent，循环工具调用直到 reasoning/fallback
    final_state = app.invoke({
        "city": request.city,
        "dates": dates,
        "days": len(dates),
        "messages": [],
        "retry_count": 0,
        "is_fallback": False,
        "step": 0,
        "pois": [],
        "weather": [],
        "clusters": {},
        "daily_plans": [],
        "tool_errors": [],
    })

    logger.info("Graph 执行结束：is_fallback=%s, 最终消息数=%d",
                final_state.get("is_fallback"),
                len(final_state.get("messages", [])))

    # 从最后一条 assistant 消息中提取 JSON
    last = final_state["messages"][-1]
    content = getattr(last, "content", "") or ""

    payload = _extract_json_payload(content)
    if payload is not None:
        logger.info("plan_trip 成功解析 JSON 输出（顶层字段：%s）", list(payload.keys()))
    else:
        logger.warning("plan_trip 无法解析为 JSON，将用 summary 兜底返回完整结构")

    raw_text = content if isinstance(content, str) else str(content)
    # 无论解析结果如何，_build_trip_response 都会返回一个合法完整的 TripResponse，
    # 不会再因缺字段抛 ValidationError 导致 /plan 返回 500。
    return _build_trip_response(payload, request, dates, raw_text)
