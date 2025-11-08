# 3.8 Mini Lab – Tool-Using Agent (Alpha Vantage)
# Uses free demo key; generate a new one with URL - https://www.alphavantage.co/support/#api-key

import requests
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict, total=False):
    symbol: str
    stock_raw: str
    summary: str
    final_answer: str
    reflection: str
    error: Optional[str]


alpha_vantage_demo_key = "api-key-here" # replace with your own for real use
# 1) Fetch stock price
def fetch_stock_node(state: AgentState) -> AgentState:
    symbol = state.get("symbol", "AAPL")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={alpha_vantage_demo_key}"
    try:
        data = requests.get(url, timeout=10).json()
        price = data.get("Global Quote", {}).get("05. price", None)
        if price:
            state["stock_raw"] = f"{symbol.upper()} current price: ${price}"
        else:
            raise ValueError("No price found in API response")
    except Exception as e:
        # fallback value so the lab always runs
        state["stock_raw"] = f"{symbol.upper()} current price: $190.75 (sample)"
        state["error"] = f"live fetch failed, using sample: {e}"
    return state

# 2) Summarize (light logic; no LLM needed for demo)
def summarize_node(state: AgentState) -> AgentState:
    stock_info = state.get("stock_raw", "No stock data.")
    state["summary"] = f"Summary: {stock_info} — the market appears stable today."
    return state

# 3) Compose output
def compose_node(state: AgentState) -> AgentState:
    state["final_answer"] = f"{state['stock_raw']}\n{state['summary']}"
    return state

# 4) Reflection
def reflection_node(state: AgentState) -> AgentState:
    ans = state.get("final_answer", "")
    if "current price" in ans and "Summary:" in ans:
        state["reflection"] = "Response looks complete ✅"
    else:
        state["reflection"] = "Response seems partial ❗"
    return state

# 5) Build LangGraph
graph = StateGraph(AgentState)
graph.add_node("fetch_stock", fetch_stock_node)
graph.add_node("summarize", summarize_node)
graph.add_node("compose", compose_node)
graph.add_node("reflect", reflection_node)

graph.add_edge(START, "fetch_stock")
graph.add_edge("fetch_stock", "summarize")
graph.add_edge("summarize", "compose")
graph.add_edge("compose", "reflect")
graph.add_edge("reflect", END)

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({"symbol": "AAPL"})
    print(result["final_answer"])
    print(result["reflection"])
    if result.get("error"):
        print("Note:", result["error"])
