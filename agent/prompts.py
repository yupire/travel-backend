from langchain_core.prompts import ChatPromptTemplate

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
