"""LangGraph 旅行规划 Agent

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
from agent.prompts import FORMAT_OUTPUT_PROMPT, build_initial_messages
from agent.fallback_graph import run_fallback


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

# 格式化链：不绑工具、开启 JSON 模式，专门把 Agent 收集到的数据整理成严格
# TripResponse JSON。DeepSeek 倾向于输出 Markdown 攻略，靠 system prompt 约束不可靠，
# 这里用 response_format=json_object 强制结构化输出。
_formatter_llm = ChatOpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-v4-pro",
    streaming=False,
    model_kwargs={"response_format": {"type": "json_object"}},
)

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


def format_node(state: TravelState) -> dict:
    """节点 4：Agent 推理结束后，把对话中收集到的全部数据整理成严格 TripResponse JSON。

    Agent（绑定工具的主链）倾向于输出 Markdown 攻略而非结构化 JSON，导致 itinerary
    解析为空。这里用开启 JSON 模式的 _formatter_llm，带着完整对话历史（含工具返回的
    经纬度/门票/交通）再跑一次，强制产出可被 TripResponse 解析的 JSON。
    """
    step = state.get("step", 0) + 1
    messages = state.get("messages") or []
    logger.info("【第 %d 步】进入 format_node[结构化输出]：city=%s, 历史消息数=%d",
                step, state.get("city"), len(messages))

    # 在完整历史后追加格式化指令，让 LLM 基于已收集数据输出严格 JSON
    convo = list(messages) + [HumanMessage(content=FORMAT_OUTPUT_PROMPT)]
    try:
        response = _formatter_llm.invoke(convo)
        logger.info("【第 %d 步】format_node 完成结构化 JSON 输出", step)
        return {"messages": [response], "step": step}
    except Exception as e:
        # 格式化失败不致命：保留 Agent 原始输出，由 plan_trip 兜底成完整结构
        logger.exception("【第 %d 步】format_node 失败，保留原始输出兜底：%r", step, e)
        return {"step": step}


def fallback_node(state: TravelState) -> dict:
    """降级节点：Agent 主链失败时，转入固定（确定性）子图重新规划。

    不再依赖 Agent 自主决策，而是调用 agent/fallback_graph 里的固定图：
    并行查询景点/天气 → 地理聚类与室内外分类 → 按天气分配 → 路线规划 →
    生成理由 → 结构化输出。子图保证产出一份合法 TripResponse；这里把它序列化成
    JSON 文本作为最后一条消息，交由下游 _finalize / _build_trip_response 解析收尾。
    """
    step = state.get("step", 0) + 1
    dates = state.get("dates") or []
    logger.warning("【第 %d 步】进入 fallback_node[固定降级子图]：city=%s, 天数=%d",
                   step, state.get("city"), state.get("days", len(dates)))

    resp = run_fallback(
        city=state.get("city", ""),
        dates=dates,
        days=state.get("days", len(dates)),
    )

    # 用标准 JSON 序列化（注意不能用 str(dict)：那是 Python repr，单引号无法被
    # json.loads 解析，会导致 _finalize 丢掉整张行程）。
    import json
    payload = json.dumps(resp.model_dump(), ensure_ascii=False)
    logger.warning("【第 %d 步】fallback_node 固定子图完成：共 %d 天行程",
                   step, len(resp.itinerary))
    return {
        "is_fallback": True,
        "messages": [SystemMessage(content=payload)],
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
    - "format"       → LLM 不再调用工具，进入格式化节点产出严格 JSON
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

    # 4) LLM 不再调用工具：进入 format 节点，用 JSON 模式把已收集数据整理成严格
    #    TripResponse JSON（Agent 主链常输出 Markdown，不能直接当最终结果）。
    logger.info("should_continue: LLM 无工具调用 -> format")
    return "format"


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

# format 与 fallback 完即结束
graph.add_edge("format", END)
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


# ──────────────────────── 公共辅助 ────────────────────────
def _expand_dates(request: TripRequest) -> list[str]:
    """把起止日期展开成逐日列表（含首尾），供各节点按天处理。"""
    from datetime import datetime, timedelta

    start = datetime.fromisoformat(request.start_date)
    end = datetime.fromisoformat(request.end_date)
    return [
        (start + timedelta(days=i)).date().isoformat()
        for i in range((end - start).days + 1)
    ]


def _initial_state(request: TripRequest, dates: list[str]) -> dict:
    """构造 Graph 入口 state。"""
    return {
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
    }


def _finalize(final_state: dict, request: TripRequest, dates: list[str]) -> TripResponse:
    """从 Graph 最终 state 中提取最后一条消息并收敛成合法 TripResponse。"""
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


def _is_reasoning_done(message) -> bool:
    """判断一条消息是否代表「Agent 推理完成、即将进入 format 结构化」。

    路由规则（见 should_continue）：当最后一条是 AI 消息且不带 tool_calls 时，
    Graph 会走向 format 节点。此刻可对外发出「规划已完成，正在整理」的进度信号，
    让前端先反馈，再等 format 结构化结果返回。
    异常 / 降级路径产出的是 SystemMessage（type=system），不会命中此判断。
    """
    if getattr(message, "type", "") != "ai":
        return False
    return not (getattr(message, "tool_calls", None) or [])


# ──────────────────────── 对外入口 ────────────────────────
def plan_trip(request: TripRequest) -> TripResponse:
    """调用 Agent，串行触发整个 Graph，返回结构化 TripResponse。"""
    logger.info("plan_trip 开始：city=%s, %s ~ %s",
                request.city, request.start_date, request.end_date)

    dates = _expand_dates(request)
    logger.info("plan_trip 行程共 %d 天", len(dates))

    # 同步触发整个 Graph：从 START -> agent，循环工具调用直到 reasoning/fallback
    final_state = app.invoke(_initial_state(request, dates))

    return _finalize(final_state, request, dates)


def plan_trip_stream(request: TripRequest):
    """流式触发 Graph：先吐进度事件，结构化完成后再吐最终结果。

    yield 的事件均为 dict：
    - {"type": "progress", "stage": "formatting", "message": ...}
        Agent 推理结束、即将进入 format 结构化时发出，前端可提示「正在整理」。
    - {"type": "result", "data": <TripResponse dict>}
        format 结构化完成后发出，携带最终行程。

    用 app.stream(stream_mode="values") 逐步拿到完整 state：每跑完一个节点
    都会吐一份完整 state，借此在「推理完成」与「结构化完成」之间插入进度信号。
    """
    logger.info("plan_trip_stream 开始：city=%s, %s ~ %s",
                request.city, request.start_date, request.end_date)

    dates = _expand_dates(request)
    logger.info("plan_trip_stream 行程共 %d 天", len(dates))

    final_state: dict | None = None
    emitted_formatting = False

    # stream_mode="values"：每步产出完整累计 state（messages 已按 operator.add 合并）
    for state in app.stream(_initial_state(request, dates), stream_mode="values"):
        final_state = state
        messages = state.get("messages") or []
        last = messages[-1] if messages else None

        # 推理完成（AI 消息且无 tool_calls）→ 下一步是 format，提前发进度信号
        if not emitted_formatting and last is not None and _is_reasoning_done(last):
            emitted_formatting = True
            logger.info("plan_trip_stream：Agent 推理完成，发出『正在整理』进度信号")
            yield {
                "type": "progress",
                "stage": "formatting",
                "message": "规划已完成，正在整理行程…",
            }

    if final_state is None:
        logger.error("plan_trip_stream：Graph 未产出任何 state")
        raise RuntimeError("规划流程未产出结果")

    resp = _finalize(final_state, request, dates)
    yield {"type": "result", "data": resp.model_dump()}
