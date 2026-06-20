"""Agent 使用的大模型实例

集中构造两条 DeepSeek（OpenAI 兼容协议）链，供 agent/nodes.py 的各节点调用：
- _llm           : 绑定工具的主链，负责推理 + 决定调用哪些工具
- _formatter_llm : 结构化输出格式化链，直接产出已校验的 TripResponse 对象

单独成模块是为了让节点逻辑与「模型如何配置」解耦：调参（超时/重试/思考开关）
只改这里，不动节点代码。
"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from models import TripResponse
from tools import ALL_TOOLS


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
    # 单次请求超时：DeepSeek 慢/挂起时抛 APITimeoutError，而非无限阻塞。
    # max_retries：超时/限流/5xx 时由 SDK 自动重试，仍失败才抛给 agent_node 兜底。
    timeout=90,
    max_retries=2,
    # 透传 OpenAI SDK 不直接支持的字段（如 thinking）
    model_kwargs={"extra_body": {"thinking": {"type": "enabled"}}},
).bind_tools(ALL_TOOLS)

# 格式化链：专门把 Agent 收集到的数据整理成严格 TripResponse。DeepSeek 倾向于输出
# Markdown 攻略，靠 system prompt 约束不可靠；这里用 with_structured_output 让 invoke
# 直接返回已按 TripResponse 校验的对象，省去从 Markdown / 脏文本里抠 JSON 的环节。
#
# method="json_mode"（即 response_format=json_object，旧代码同款，与 DeepSeek 的
# thinking 模式兼容）：deepseek-v4-pro 默认开 thinking，而 function_calling / json_schema
# 会强制 tool_choice，thinking 模式不支持 → 400「Thinking mode does not support this
# tool_choice」。json_mode 不走 tool_choice，只强制输出合法 JSON，再由 LangChain 解析
# 成 TripResponse（schema 不合法则抛 ValidationError，交给 format_node 节点级重试）。
# 注意：json_mode 要求 prompt 里出现 "json" 字样——FORMAT_OUTPUT_PROMPT 已满足。
_formatter_llm = ChatOpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-v4-pro",
    streaming=False,
    # 结构化整理要重读整段对话历史，单次较慢，给足超时；超时由 SDK 重试一次，
    # 仍失败则抛出，交给 format_node 的节点级重试（见 after_format）。
    timeout=120,
    max_retries=1,
).with_structured_output(TripResponse, method="json_mode")
