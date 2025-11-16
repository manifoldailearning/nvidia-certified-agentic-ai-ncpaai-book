from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# -----------------------------------------
# 1. Define the state
# -----------------------------------------
class State(TypedDict, total=False):
    task: str
    proposal: str
    decision: str


# -----------------------------------------
# 2. AI node: generate a proposed action
# -----------------------------------------
def ai_generate(state: State) -> State:
    task = state["task"]
    proposal = f"Proposed action: {task}"
    print(f"[AI] {proposal}")
    return {
        **state,
        "proposal": proposal,
    }


# -----------------------------------------
# 3. Human approval node
# -----------------------------------------
def human_approval(state: State) -> State:
    task_output = state["proposal"]
    print(f"[HUMAN REVIEW] Review required: {task_output}")
    approval = input("Approve this action? (yes/no): ")
    decision = "Approved" if approval.strip().lower() == "yes" else "Rejected"
    print(f"[DECISION] {decision}")
    return {
        **state,
        "decision": decision,
    }


# -----------------------------------------
# 4. Build the graph
# -----------------------------------------
graph = StateGraph(State)
graph.add_node("generator", ai_generate)
graph.add_node("review", human_approval)

graph.add_edge(START, "generator")
graph.add_edge("generator", "review")
graph.add_edge("review", END)

app = graph.compile()

# -----------------------------------------
# 5. Run the workflow
# -----------------------------------------
if __name__ == "__main__":
    result = app.invoke({"task": "Update customer credit limit"})
    print("\n=== Final State ===")
    print(result)
    print("Final decision:", result.get("decision"))
