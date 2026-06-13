from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from typing import TypedDict, Annotated
import operator

# 定义state和工具接口

# State 是贯穿整个 graph 的上下文
class TravelState(TypedDict):
    city: str
    dates: list[str]          # ["2024-03-01", "2024-03-03"]
    days: int
    pois: list[dict]          # 高德 POI 结果
    weather: list[dict]       # 和风天气每日数据
    clusters: dict            # {cluster_id: [poi_list]}
    daily_plan: list[dict]    # 每日草稿行程
    messages: Annotated[list, operator.add]  # Agent 的完整对话历史
    final_itinerary: dict     # 最终输出

# 注册工具 —— LLM 靠这些描述来决定调用哪个
@tool
def search_pois(city: str, category: str = "景点") -> list[dict]:
    """搜索城市内的景点 POI，返回名称、坐标、类型、评分。
    category 可选：景点、博物馆、公园、购物、餐厅"""
    # 调用高德 POI 搜索 API
    ...

@tool  
def get_weather(city_id: str, dates: list[str]) -> list[dict]:
    """查询指定城市在出行日期内每天的天气，返回天气状况、降水概率、温度。
    city_id 从和风天气城市 API 获取"""
    ...

@tool
def cluster_pois(pois: list[dict], n_clusters: int) -> dict:
    """用 KMeans 对景点按地理位置聚类，返回每个 cluster 的景点列表。
    n_clusters 建议等于出行天数"""
    ...

@tool
def get_route(origin: dict, destination: dict, waypoints: list[dict]) -> dict:
    """调用高德路径规划，返回景点间的交通方式、时间、距离"""
    ...