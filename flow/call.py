from langgraph.graph import StateGraph, END
from tool import TravelState
from node import should_continue, agent_node, tool_node

# 组件graph并运

# 构建 graph
graph = StateGraph(TravelState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)
graph.add_edge("tools", "agent")  # 工具结果回到 agent 继续推理

app = graph.compile()

# 调用入口
result = app.invoke({
    "city": "北京",
    "dates": ["2024-03-01", "2024-03-02", "2024-03-03"],
    "days": 3,
    "messages": []
})