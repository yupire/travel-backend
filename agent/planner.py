"""旅行规划 Agent —— 对外入口

本模块是整个 Agent 的对外门面，对外暴露两个入口：
- plan_trip        : 同步触发整个 Graph，返回结构化 TripResponse
- plan_trip_stream : 流式触发，逐步吐进度事件，最后吐最终结果

Graph 的具体结构与节点逻辑已拆分到同包其它模块：
- agent/llm.py              : 两条 DeepSeek 大模型链（主链 + JSON 格式化链）
- agent/state.py            : TravelState 共享状态定义
- agent/nodes.py            : agent / tools / format / fallback 四个节点
- agent/graph.py            : 路由决策（should_continue / after_format）与 Graph 组装（app）
- agent/response_builder.py : 输出解析 + 逐字段兜底（保证返回合法 TripResponse）
- agent/progress.py         : 流式进度文案
"""
from __future__ import annotations

import logging
import queue
import threading

from models import TripRequest, TripResponse
from agent.graph import app
from agent.response_builder import _extract_json_payload, _build_trip_response
from agent.progress import _describe_tool_call

logger = logging.getLogger(__name__)

# LangGraph 节点遍历的硬上限（默认 25）。真正控制循环规模的是 graph.should_continue 里
# 的 MAX_TOOL_ROUNDS 软上限——正常应在这之前就强制收尾。这里给一个明显大于软上限路径
# （8 轮 agent+tools≈16 + format 重试等）的值作为硬兜底，确保软上限先生效、不至于因撞上
# 默认 25 抛 GraphRecursionError；万一逻辑异常导致失控，也会在 50 步处止血。
_GRAPH_CONFIG = {"recursion_limit": 50}


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
        "tool_fail_counts": {},
    }


def _finalize(final_state: dict, request: TripRequest, dates: list[str]) -> TripResponse:
    """从 Graph 最终 state 中提取最后一条消息并收敛成合法 TripResponse。"""
    logger.info("Graph 执行结束：is_fallback=%s, 最终消息数=%d",
                final_state.get("is_fallback"),
                len(final_state.get("messages", [])))

    # 从最后一条 assistant 消息中提取 JSON。messages 为空时不抛 IndexError，
    # 走 payload=None 的兜底分支，仍返回结构完整（itinerary=[]）的 TripResponse。
    messages = final_state.get("messages") or []
    last = messages[-1] if messages else None
    if last is None:
        logger.warning("_finalize：最终 state 无任何消息，将返回空行程兜底结构")
    content = getattr(last, "content", "") or ""

    payload = _extract_json_payload(content)
    if payload is not None:
        raw_it = payload.get("itinerary")
        logger.info(
            "plan_trip 成功解析 JSON 输出（顶层字段：%s；itinerary 类型=%s，原始天数=%s）",
            list(payload.keys()),
            type(raw_it).__name__,
            len(raw_it) if isinstance(raw_it, list) else "N/A",
        )
    else:
        logger.warning("plan_trip 无法解析为 JSON，将用 summary 兜底返回完整结构")

    raw_text = content if isinstance(content, str) else str(content)
    # 无论解析结果如何，_build_trip_response 都会返回一个合法完整的 TripResponse，
    # 不会再因缺字段抛 ValidationError 导致 /plan 返回 500。
    return _build_trip_response(payload, request, dates, raw_text)


# ──────────────────────── 对外入口 ────────────────────────
def plan_trip(request: TripRequest) -> TripResponse:
    """调用 Agent，串行触发整个 Graph，返回结构化 TripResponse。"""
    logger.info("plan_trip 开始：city=%s, %s ~ %s",
                request.city, request.start_date, request.end_date)

    dates = _expand_dates(request)
    logger.info("plan_trip 行程共 %d 天", len(dates))

    # 同步触发整个 Graph：从 START -> agent，循环工具调用直到 reasoning/fallback
    final_state = app.invoke(_initial_state(request, dates), config=_GRAPH_CONFIG)

    return _finalize(final_state, request, dates)


def _plan_events(request: TripRequest):
    """流式触发 Graph：每一步都吐进度事件，结构化完成后再吐最终结果。

    yield 的事件均为 dict：
    - {"type": "progress", "step": N, "stage": "tool", "message": ...}
        Agent 每决定调用一个工具就发一条，前端按步骤列表展示「当前在做什么」。
    - {"type": "progress", "step": N, "stage": "formatting", "message": ...}
        Agent 推理结束、即将进入 format 结构化时发出，提示「正在整理」。
    - {"type": "result", "data": <TripResponse dict>}
        format 结构化完成后发出，携带最终行程；前端收到后清空进度、只展示行程。

    注意：本函数是「纯事件源」，不负责保活；format_node 是一次可能耗时数十秒的阻塞
    LLM 调用，期间这里不会 yield 任何事件。保活心跳由外层 plan_trip_stream 注入。

    用 app.stream(stream_mode="values") 逐步拿到完整 state：每跑完一个节点都会吐一份
    完整累计 state。这里跟踪「已处理到第几条消息」，对每条新增的 AI 消息按其 tool_calls
    展开成进度，做到「每一步调用工具都给前端一条进度」。
    """
    logger.info("plan_trip_stream 开始：city=%s, %s ~ %s",
                request.city, request.start_date, request.end_date)

    dates = _expand_dates(request)
    logger.info("plan_trip_stream 行程共 %d 天", len(dates))

    final_state: dict | None = None
    seen = 0                 # 已处理（已转成进度）的 messages 数量
    emitted_formatting = False

    # 起手先发一条，告诉用户 Agent 已经开始分析需求
    yield {
        "type": "progress",
        "step": 0,
        "stage": "start",
        "message": "AI 正在分析行程需求…",
    }

    # stream_mode="values"：每步产出完整累计 state（messages 已按 operator.add 合并）
    for state in app.stream(_initial_state(request, dates), stream_mode="values",
                            config=_GRAPH_CONFIG):
        final_state = state
        messages = state.get("messages") or []
        step = state.get("step", 0)

        # 只处理本步新增的消息，避免重复发送历史消息对应的进度
        new_messages = messages[seen:]
        seen = len(messages)

        for msg in new_messages:
            if getattr(msg, "type", "") != "ai":
                continue
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                # AI 决定调用工具：逐个工具发一条进度（含友好动作 + 关键参数）
                for tc in tool_calls:
                    desc = _describe_tool_call(tc)
                    logger.info("plan_trip_stream：第 %d 步进度 → %s", step, desc)
                    yield {
                        "type": "progress",
                        "step": step,
                        "stage": "tool",
                        "message": desc,
                    }
            elif not emitted_formatting:
                # AI 不再调用工具 → 下一步是 format，发『正在整理』进度信号
                emitted_formatting = True
                logger.info("plan_trip_stream：Agent 推理完成，发出『正在整理』进度信号")
                yield {
                    "type": "progress",
                    "step": step,
                    "stage": "formatting",
                    "message": "规划已完成，正在整理行程…",
                }

    if final_state is None:
        logger.error("plan_trip_stream：Graph 未产出任何 state")
        raise RuntimeError("规划流程未产出结果")

    # result 事件是前端渲染行程的唯一依据，必须保证吐出：即便 _finalize 因任何
    # 意外异常失败，也兜底返回一个结构完整（itinerary=[]）的 TripResponse，
    # 而不是让前端落到 error 分支、整页空白。
    try:
        resp = _finalize(final_state, request, dates)
    except Exception as e:
        logger.exception("_finalize 意外失败，返回空行程兜底结构：%r", e)
        resp = TripResponse(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            total_days=len(dates),
            itinerary=[],
            summary="（行程生成失败，请重试）",
        )
    yield {"type": "result", "data": resp.model_dump()}


# 阻塞步骤（format_node 的 LLM 调用）期间多久没有真实事件就补一条心跳。取值要明显
# 小于浏览器/反向代理的空闲超时（通常 30~60s+），5s 足够保活且开销极低。
_HEARTBEAT_INTERVAL = 5


def plan_trip_stream(request: TripRequest):
    """对外的流式入口：在 _plan_events 的真实事件之间注入心跳，避免 SSE 长时间静默。

    _plan_events 里的 format_node 是一次可能耗时数十秒的阻塞 LLM 调用，期间不产生任何
    事件。若直接把它当生成器 yield，这段时间连接一个字节都不发 → 被浏览器/反代判定空闲
    超时而断开（前端表现为 network error）。这里把 _plan_events 放到后台线程推进队列，
    主协程用带超时的 get 抽事件：
    - 正常拿到事件          -> 原样透传（progress / result）；
    - 超时（队列静默）      -> 说明正卡在阻塞步骤，补一条 {"type":"heartbeat"} 保活；
    - 线程内抛异常          -> 原样重抛，交由 router 的 event_stream 兜底成 error 事件。

    心跳事件对前端无业务含义，前端按「未知 type」忽略即可。
    """
    q: "queue.Queue[tuple]" = queue.Queue()
    _DONE = object()

    def _worker():
        try:
            for event in _plan_events(request):
                q.put(("event", event))
        except Exception as e:  # noqa: BLE001 —— 原样转交主协程，不在子线程里吞掉
            q.put(("error", e))
        finally:
            q.put((_DONE, None))

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        try:
            kind, payload = q.get(timeout=_HEARTBEAT_INTERVAL)
        except queue.Empty:
            # 队列静默：正处于 format 等阻塞步骤，补心跳维持连接活性
            yield {"type": "heartbeat"}
            continue
        if kind is _DONE:
            break
        if kind == "error":
            raise payload
        yield payload
