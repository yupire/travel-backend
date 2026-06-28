"""Graph 节点定义

四个节点对应 Graph 的四个处理阶段（整体流程图见 agent/graph.py 的模块文档）：
- agent_node    : 调用绑定工具的 LLM 推理，决定调用哪些工具 / 是否结束
- tools_node    : 包装 prebuilt ToolNode，执行工具并收集失败、回灌引导消息
- format_node   : 用结构化输出 LLM 把对话数据整理成已校验的 TripResponse
- fallback_node : 主链失败时转入固定降级子图重新规划
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from tools import ALL_TOOLS
from agent.prompts import FORMAT_OUTPUT_PROMPT, build_initial_messages
from agent.fallback_graph import run_fallback
from agent.llm import _llm, _formatter_llm
from agent.state import TravelState

logger = logging.getLogger(__name__)


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


# 同一个工具跨轮失败到这个次数后就「放弃」：不再要求 LLM 修复重试，改为提示它跳过
# 该数据 / 用合理默认值继续。挡住对本质拿不到的数据（如超出预报窗口的天气、不支持的
# 城市）反复打同一个接口造成的死循环式重复调用。
MAX_TOOL_FAIL = 2


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

    # ToolNode 返回 {"messages": [ToolMessage, ...]}，逐条记录工具产出。
    # fail_counts 跨轮累计每个工具的失败次数（上一轮的计数从 state 取出来续累）。
    new_msgs = result.get("messages", []) if isinstance(result, dict) else []
    fail_counts = dict(state.get("tool_fail_counts") or {})
    tool_errors: list[dict] = []
    for m in new_msgs:
        content = str(getattr(m, "content", ""))
        preview = content if len(content) <= 200 else content[:200] + "…"
        name = getattr(m, "name", "?")
        logger.info("【第 %d 步】  └─ 工具 %s 返回：内容长度=%d，预览=%s",
                    step, name, len(content), preview)
        if _is_failed_tool_message(m, content):
            fail_counts[name] = fail_counts.get(name, 0) + 1
            tool_errors.append({
                "tool": name,
                "detail": content[:300],
                "fails": fail_counts[name],
            })
            logger.warning("【第 %d 步】  ⚠ 工具 %s 调用失败（累计 %d 次）：%s",
                           step, name, fail_counts[name], content[:200])

    out = dict(result) if isinstance(result, dict) else {"messages": new_msgs}
    out["step"] = step
    out["tool_errors"] = tool_errors
    out["tool_fail_counts"] = fail_counts

    # 把失败的工具调用按「累计失败次数」分流，再回灌一条引导消息：
    # - retryable（< 阈值）：要求修正参数后重试，拿到数据前不要输出最终行程；
    # - give_up（≥ 阈值）：明确告诉 LLM 别再重复调用，跳过该数据 / 用合理默认值继续，
    #   避免对本质拿不到的数据反复打同一个接口造成死循环式重复调用。
    if tool_errors:
        retryable = [e for e in tool_errors if e["fails"] < MAX_TOOL_FAIL]
        give_up = [e for e in tool_errors if e["fails"] >= MAX_TOOL_FAIL]
        sections: list[str] = []
        if retryable:
            names = "、".join(sorted({e["tool"] for e in retryable}))
            detail_lines = "\n".join(f"- {e['tool']}: {e['detail']}" for e in retryable)
            sections.append(
                f"⚠ 以下 {len(retryable)} 个工具调用失败（{names}），请检查并修正参数后重试"
                f"（例如天气查询失败时换用城市名或确认日期在预报窗口内）。在成功拿到这些"
                f"关键数据之前，不要直接输出最终行程 JSON：\n{detail_lines}"
            )
        if give_up:
            names = "、".join(sorted({e["tool"] for e in give_up}))
            detail_lines = "\n".join(
                f"- {e['tool']}（已失败 {e['fails']} 次）: {e['detail']}" for e in give_up
            )
            sections.append(
                f"🛑 以下工具已多次失败（{names}），请不要再重复调用它们，改为跳过该部分"
                f"数据或使用合理默认值，基于已经拿到的数据继续完成规划：\n{detail_lines}"
            )
        out["messages"] = list(new_msgs) + [HumanMessage(content="\n\n".join(sections))]
    return out


def format_node(state: TravelState) -> dict:
    """节点 4：Agent 推理结束后，把对话中收集到的全部数据整理成严格 TripResponse JSON。

    Agent（绑定工具的主链）倾向于输出 Markdown 攻略而非结构化 JSON，导致 itinerary
    解析为空。这里用结构化输出的 _formatter_llm，带着完整对话历史（含工具返回的
    经纬度/门票/交通）再跑一次：invoke 直接返回已按 TripResponse schema 校验的对象，
    再序列化成干净 JSON 文本（与 fallback_node 同一套路），下游 _finalize 无需从
    Markdown 里抠 JSON。
    """
    step = state.get("step", 0) + 1
    fmt_retry = state.get("format_retry", 0)
    messages = state.get("messages") or []
    logger.info("【第 %d 步】进入 format_node[结构化输出]：city=%s, 历史消息数=%d, 当前格式化重试=%d",
                step, state.get("city"), len(messages), fmt_retry)

    # 在完整历史后追加格式化指令，让 LLM 基于已收集数据按 schema 产出结构化结果
    convo = list(messages) + [HumanMessage(content=FORMAT_OUTPUT_PROMPT)]
    try:
        resp = _formatter_llm.invoke(convo)  # 直接得到已校验的 TripResponse
        payload = json.dumps(resp.model_dump(), ensure_ascii=False)
        logger.info("【第 %d 步】format_node 完成结构化输出：共 %d 天行程",
                    step, len(resp.itinerary))
        return {"messages": [SystemMessage(content=payload)], "step": step, "format_done": True}
    except Exception as e:
        # 失败（含超时）不致命：累加重试计数，由 after_format 决定再跑一次还是结束兜底
        logger.exception("【第 %d 步】format_node 失败，format_retry -> %d：%r",
                         step, fmt_retry + 1, e)
        return {"step": step, "format_retry": fmt_retry + 1}


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
    payload = json.dumps(resp.model_dump(), ensure_ascii=False)
    logger.warning("【第 %d 步】fallback_node 固定子图完成：共 %d 天行程",
                   step, len(resp.itinerary))
    return {
        "is_fallback": True,
        "messages": [SystemMessage(content=payload)],
        "step": step,
    }
