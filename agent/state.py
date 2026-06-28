"""Graph 状态定义

TravelState 是贯穿整个 LangGraph 的共享状态，所有节点读写它。单独成模块以便
nodes / graph / planner 各处统一引用，避免循环依赖。
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


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
    - tool_fail_counts: 按工具名跨轮累计的失败次数，用于「同一工具失败到阈值后放弃、
      不再要求 LLM 修复重试」，避免对本质拿不到的数据反复打同一个接口
    - format_retry: format_node 因超时/异常失败的重试次数
    - format_done: format_node 是否已成功产出结构化 JSON（路由判定是否结束）
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
    tool_fail_counts: dict
    format_retry: int
    format_done: bool
