# 3.8 Mini Lab – Tool-Using Agent (Alpha Vantage + Gemini 2.5 Flash)
# Get a free API key here: https://www.alphavantage.co/support/#api-key
# Add your Google API key to .env as GOOGLE_API_KEY

import requests
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM for summarization
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

class AgentState(TypedDict, total=False):
    symbol: str
    stock_raw: str
    summary: str
    final_answer: str
    reflection: str
    error: Optional[str]

# replace with your own Alpha Vantage key
alpha_vantage_api_key = "api-key-here"

# 1) Fetch stock price from Alpha Vantage
def fetch_stock_node(state: AgentState) -> AgentState:
    symbol = state.get("symbol", "AAPL")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={alpha_vantage_api_key}"
    )
    try:
        data = requests.get(url, timeout=10).json()
        price = data.get("Global Quote", {}).get("05. price", None)
        if price:
            state["stock_raw"] = f"{symbol.upper()} current price: ${price}"
        else:
            raise ValueError("No price found in API response")
    except Exception as e:
        # fallback so lab always works
        state["stock_raw"] = f"{symbol.upper()} current price: $190.75 (sample)"
        state["error"] = f"live fetch failed, using sample: {e}"
    return state

# 2) Summarize with LLM
def summarize_node(state: AgentState) -> AgentState:
    stock_info = state.get("stock_raw", "No stock data.")
    prompt = (
        "You are a financial assistant. Based on the following stock info, "
        "write a short, user-friendly market summary in 2 sentences.\n\n"
        f"Stock info: {stock_info}"
    )
    try:
        if GOOGLE_API_KEY:
            res = llm.invoke(prompt)
            # langchain-google-genai returns an object with .content
            state["summary"] = res.content if hasattr(res, "content") else str(res)
        else:
            # fallback if LLM key not set
            state["summary"] = f"Summary: {stock_info} — the market appears stable today."
    except Exception as e:
        state["summary"] = "Summary unavailable due to LLM error."
        prior_err = state.get("error", "")
        state["error"] = (prior_err + f" | llm failed: {e}").strip()
    return state

# 3) Compose final output
def compose_node(state: AgentState) -> AgentState:
    stock = state.get("stock_raw", "")
    summary = state.get("summary", "")
    state["final_answer"] = f"{stock}\n\nMarket summary:\n{summary}"
    return state

# 4) Reflection / completeness check
def reflection_node(state: AgentState) -> AgentState:
    ans = state.get("final_answer", "")
    has_stock = "current price" in ans
    has_summary = "Market summary" in ans or "Summary:" in ans
    if has_stock and has_summary:
        state["reflection"] = "Response looks complete ✅"
    else:
        state["reflection"] = "Response seems partial ❗"
    return state

# 5) Build graph
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
    print(result.get("final_answer", ""))
    print(result.get("reflection", ""))
    if result.get("error"):
        print("Note:", result["error"])
