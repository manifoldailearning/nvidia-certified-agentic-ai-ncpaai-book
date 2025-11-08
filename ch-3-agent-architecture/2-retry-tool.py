from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1) shared state
class State(TypedDict, total=False):
    attempts: int
    message: str
    status: str  # "ok" or "error"

# 2) node that MAY fail, but we catch it inside
def risky_step(state: State) -> State:
    attempts = state.get("attempts", 0) + 1

    try:
        # simulate flaky tool
        if attempts < 3:
            raise Exception(f"Simulated failure on attempt {attempts}")
        # success path
        return {
            "attempts": attempts,
            "message": "Task succeeded after retries ",
            "status": "ok",
        }
    except Exception as e:
        # soft-fail: return state instead of raising
        return {
            "attempts": attempts,
            "message": str(e),
            "status": "error",
        }

# 3) router: where do we go next?
def decide_next(state: State) -> str:
    # retry up to 3 attempts
    if state.get("status") == "error" and state.get("attempts", 0) < 3:
        return "risky"   # go back and try again
    return "end"         # stop the workflow

# 4) build graph
graph = StateGraph(State)
graph.add_node("risky", risky_step)
graph.add_edge(START, "risky")

graph.add_conditional_edges(
    "risky",
    decide_next,
    {
        "risky": "risky",
        "end": END,
    },
)

app = graph.compile()

# 5) run once
result = app.invoke({"attempts": 0, "message": "", "status": "ok"})
print(result)
