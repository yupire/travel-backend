from langchain_anthropic import ChatAnthropic
from tool import search_pois, get_weather, cluster_pois, get_route, TravelState
from langchain_core.messages import SystemMessage, HumanMessage

# 定义graph节点和路由逻辑

llm = ChatAnthropic(model="claude-opus-4-5").bind_tools(
    [search_pois, get_weather, cluster_pois, get_route]
)

SYSTEM_PROMPT = """你是一个旅行规划专家 Agent。你有以下工具：
- search_pois：搜索城市景点
- get_weather：查询出行日期天气
- cluster_pois：将景点按地理位置聚类，每天安排一个区域
- get_route：规划景点间路线

规划流程：
1. 先搜索景点，同时查天气（可并行）
2. 对景点聚类，区域数 = 出行天数
3. 按天气调整：降水概率>60% 的日期优先安排室内景点
4. 对每天的景点安排规划路线
5. 生成最终行程，每天附上天气说明和安排理由

室内/户外判断依据：
- POI type 含 110200（文化场馆）/ 110202（博物馆）= 室内
- POI type 含 110101（公园）/ 110104（广场）= 户外
- 名称含"购物中心/商场/影院" = 室内，含"山/湖/海/森林" = 户外
"""

def agent_node(state: TravelState):
    """Agent 思考节点：LLM 决定下一步调用什么工具"""
    messages = state["messages"]
    if not messages:
        # 初始化：把用户需求注入
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"帮我规划{state['city']}的行程，"
                                 f"日期：{state['dates']}，共{state['days']}天")
        ]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

def tool_node(state: TravelState):
    """工具执行节点：执行 LLM 选定的工具并把结果存回 state"""
    from langchain_core.messages import ToolMessage
    
    last_msg = state["messages"][-1]
    tool_results = []
    
    for tool_call in last_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # 执行对应工具
        result = TOOL_MAP[tool_name].invoke(tool_args)
        
        # 同时更新 state 中对应字段
        if tool_name == "search_pois":
            state["pois"] = result
        elif tool_name == "get_weather":
            state["weather"] = result
        elif tool_name == "cluster_pois":
            state["clusters"] = result
            
        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
    
    return {"messages": tool_results, **state}

def should_continue(state: TravelState) -> str:
    """路由函数：决定继续调用工具还是结束"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"   # 还有工具要调用
    return END           # LLM 没有更多工具调用 = 完成