@tool
def calculate_area(radius: float) -> float:
    """Calculates area of a circle."""
    return 3.14 * radius * radius

@tool
def wiki_search(topic: str) -> str:
    """Fetches a short summary from Wikipedia."""
    return wiki_summary(topic, sentences=3)

TOOLS = {"calculate_area": calculate_area, "wiki_search": wiki_search}

def agent_node(state: AgentState) -> AgentState:
    resp = llm.invoke(state["messages"], tools=list(TOOLS.values()))
    return {"messages": state["messages"] + [resp]}