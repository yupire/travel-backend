"""旅行规划 Agent 的所有 Prompt 集中管理。

设计原则：
- 每条 prompt 单独暴露一个常量，便于复审 / 改写 / A/B 测试；
- 配套的 ChatPromptTemplate（如有）紧跟其下；
- 不要在 planner.py 中再出现长段 prompt 字符串，统一从本模块导入。
"""
from __future__ import annotations

from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


# ──────────────────────── TripResponse JSON Schema ────────────────────────
# 同时供 Agent System Prompt 与最终格式化节点引用，避免 schema 漂移。
TRIP_JSON_SCHEMA = """{
  "city": str,
  "start_date": str,
  "end_date": str,
  "total_days": int,
  "itinerary": [
    {
      "day": int,
      "date": str,
      "weather": {"condition": str, "temp_high": int, "temp_low": int, "description": str},
      "spots": [
        {
          "id": str, "name": str, "lat": float, "lng": float,
          "duration_min": int, "open_time": str, "ticket": float,
          "type": "indoor|outdoor", "is_indoor": bool,
          "tags": list[str], "description": str,
          "nearby_foods": list[{"id": str, "name": str, "cuisine": str, "price_range": str, "rating": float, "distance_m": int}],
          "transport_from_prev": {"mode": str, "duration_min": int, "cost": float, "distance_km": float} | null
        }
      ],
      "reasoning": "2-3 句中文：解释当天的天气/景点顺序/美食搭配",
      "is_indoor_outdoor_filter": bool
    }
  ],
  "summary": "3-4 句整体旅行总结"
}"""


# ──────────────────────── Agent System Prompt ────────────────────────
SYSTEM_PROMPT = """你是一名专业的旅行规划 Agent，遵循以下流程完成一次行程规划：

1. 调用 list_supported_cities 确认城市在受支持列表中；
2. 并行调用 get_tourist_spots 和 get_city_weather 获取候选景点和天气；
3. 用 cluster_spots_geographically 把景点按地理位置分组（簇数 ≈ 出行天数）；
4. 对每个簇，用 geocode_spot_locations + classify_spot_indoor_outdoor 判断室内/户外；
5. 如果遇到雨天日期，优先安排室内景点（通过 reorder 决定访问顺序）；
6. 用 plan_route_directions 计算相邻景点的交通（mode=transit/driving/walking）；
7. 整合所有数据后输出最终 JSON。

工具调用规则：
- 每次只输出 1 个工具调用请求；
- 拿到工具结果后再决定下一步；
- 数据齐全后，用纯文本说明你的规划即可，系统会有专门步骤把它整理成 JSON。

JSON 输出 schema（最终结构参考）：
""" + TRIP_JSON_SCHEMA + """

严格要求：
- 所有景点必须能在 get_tourist_spots 的结果里找到对应 id；
- 每日的 weather 必须能匹配 get_city_weather 的实际返回；
- 景点的 lat/lng 必须来自 geocode_spot_locations 的真实返回，不可编造。
"""


# ──────────────────────── 最终格式化 Prompt ────────────────────────
# Agent 推理结束后，由 format_node 用 JSON 模式 LLM 把对话中收集到的全部数据
# （工具返回的经纬度/门票/交通 + Agent 的规划）整理成严格的 TripResponse JSON。
FORMAT_OUTPUT_PROMPT = (
    "现在把你上面已经完成的行程规划整理成严格的 JSON，供系统解析。\n\n"
    "硬性要求：\n"
    "- 只输出一个 JSON 对象，禁止任何 Markdown、表格、emoji、解释性文字或代码块标记；\n"
    "- 顶层必须包含：city, start_date, end_date, total_days, itinerary, summary；\n"
    "- itinerary 是按天数组，每天必须含 day, date, weather, spots, reasoning, is_indoor_outdoor_filter；\n"
    "- spots 每项必须含 id, name, lat, lng, duration_min, open_time, ticket, type, is_indoor, "
    "tags, description, nearby_foods, transport_from_prev；\n"
    "- lat/lng/门票/交通等数值一律使用前面工具返回的真实数据，缺失就用合理默认值（如 ticket=0、"
    "transport_from_prev=null），但绝不要凭空编造坐标；\n"
    "- weather 必须与 get_city_weather 的实际返回一致。\n\n"
    "JSON schema：\n" + TRIP_JSON_SCHEMA
)


# ──────────────────────── Reasoning Prompt（已存在，保留）────────────────────────
PLAN_REASONING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是专业旅游规划师。请用中文为每日行程生成简洁说明，并生成整体旅行总结。"
        "严格按JSON格式输出，不要有额外文字或代码块标记。",
    ),
    (
        "human",
        """城市：{city}，行程：{start_date} 至 {end_date}

每日数据：
{days_json}

输出格式（严格JSON，无其他文字）：
{{
  "days": [
    {{"day": 1, "reasoning": "2-3句中文：结合天气说明景点安排顺序，以及附近美食搭配建议"}}
  ],
  "summary": "3-4句整体旅行总结"
}}""",
    ),
])


# ──────────────────────── 工厂函数 ────────────────────────
def build_initial_messages(city: str, dates: Sequence[str], days: int) -> list[BaseMessage]:
    """构造 Agent 起始 messages：SystemMessage + 一条 HumanMessage。

    planner.py 不再直接拼接 prompt 字符串，统一调用此函数。
    """
    start_date = dates[0] if dates else ""
    end_date = dates[-1] if dates else ""
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"请帮我规划 {city} 的行程，"
                f"日期：{start_date} 至 {end_date}，"
                f"共 {days} 天。"
            )
        ),
    ]
