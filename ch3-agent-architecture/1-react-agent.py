# react_agent_graph.py
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 1) Define tool
@tool
def calculate_area(radius: float) -> float:
    """Calculates area of a circle."""
    return 3.14 * radius * radius

TOOLS = {
    "calculate_area": calculate_area,
}

# 2) Define state
class AgentState(TypedDict):
    messages: List[AnyMessage]

# 3) LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

# 4) Node: agent thinks + maybe calls tool
def agent_node(state: AgentState) -> AgentState:
    """
    Takes messages, asks LLM what to do.
    If LLM decides to call a tool, it will emit a tool-call message.
    """
    resp = llm.invoke(state["messages"], tools=list(TOOLS.values()))
    return {"messages": state["messages"] + [resp]}

# 5) Node: run tools (if any tool calls are present)
def tool_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)

    new_messages: List[AnyMessage] = state["messages"][:]

    if tool_calls:
        for call in tool_calls:
            name = call["name"]
            args = call["args"]
            tool_fn = TOOLS[name]
            result = tool_fn.invoke(args)
            new_messages.append(
                ToolMessage(
                    name=name,
                    content=str(result),
                    tool_call_id=call["id"],
                )
            )

    return {"messages": new_messages}

# 6) Conditional: do we need to go back to agent or finish?
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    # if LLM just asked for a tool -> go to tool node
    if getattr(last, "tool_calls", None):
        return "tools"
    # else we're done
    return "end"

def build_react_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # start -> agent
    graph.add_edge(START, "agent")
    # agent -> (tools or end)
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "end": END,
    })
    # after tools, go back to agent to let it read tool result
    graph.add_edge("tools", "agent")

    return graph.compile()

if __name__ == "__main__":
    app = build_react_graph()
    result = app.invoke({
        "messages": [HumanMessage(content="calculate the area of a circle with radius 10")]
    })
    print("Raw result:", result)
    # final answer is the last AI message
    final_msg = [m for m in result["messages"] if isinstance(m, AIMessage)][-1]
    print("Final answer:")
    print(final_msg.content)
