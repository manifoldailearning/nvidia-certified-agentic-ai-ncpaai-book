"""
Mini Lab – Planner → Executor → Memory → Reflection (LangGraph)

What this does:
1. User gives a goal (e.g. "Write a report on generative AI").
2. Planner node turns that into a 3-step plan.
3. Executor node asks the LLM to "execute" that plan.
4. Reflection node asks the LLM to critique the result.
5. The whole run is saved to SQLite, so re-running with the same thread_id
   will keep the previous messages/state.

Prereqs:
    pip install langgraph langgraph-checkpoint-sqlite langchain-openai
    export OPENAI_API_KEY="..."

"""

from typing import TypedDict
from typing_extensions import Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

# ----- 1. Define the agent state -----
class AgentState(TypedDict):
    goal: str
    plan: str
    result: str
    reflection: str
    # this lets LangGraph merge messages across nodes & runs
    messages: Annotated[list[HumanMessage], add_messages]

# ---- Initialize model ----
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)


# ----- 2. Nodes -----
def planner_node(state: AgentState) -> AgentState:
    goal = state["goal"]
    plan = (
        f"Plan for: {goal}\n"
        "1) Gather background/context\n"
        "2) Produce a concise summary\n"
        "3) Present as bullet points"
    )
    state["plan"] = plan

    # add to message history (optional but nice for debugging)
    return {
        **state,
        "messages": [HumanMessage(content=f"[planner] Created plan:\n{plan}")]
    }


def executor_node(state: AgentState) -> AgentState:
    plan = state["plan"]
    goal = state["goal"]

    prompt = (
        "You are an AI executor. Follow the plan below to complete the goal.\n"
        f"Goal: {goal}\n"
        f"Plan:\n{plan}\n"
        "Return the final answer in 4-6 bullet points."
    )

    resp = llm.invoke([HumanMessage(content=prompt)])
    result = resp.content
    state["result"] = result

    return {
        **state,
        "messages": [HumanMessage(content=f"[executor] Output:\n{result}")]
    }


def reflection_node(state: AgentState) -> AgentState:
    result = state["result"]

    prompt = (
        "You are a reflection module for an agent.\n"
        "Critique the following output for clarity, completeness, and usefulness.\n"
        "Then suggest ONE improvement for the next run.\n\n"
        f"Agent Output:\n{result}"
    )

    resp = llm.invoke([HumanMessage(content=prompt)])
    reflection = resp.content
    state["reflection"] = reflection

    return {
        **state,
        "messages": [HumanMessage(content=f"[reflection] {reflection}")]
    }


# ----- 3. Build the LangGraph workflow -----
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("reflection", reflection_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "reflection")
workflow.add_edge("reflection", END)

# ----- 4. Compile with SQLite checkpointer (for memory) -----
if __name__ == "__main__":
    # this file will hold the agent's state between runs
    with SqliteSaver.from_conn_string("chapter5_minilab.db") as checkpointer:
        app = workflow.compile(checkpointer=checkpointer)

        # you can change this to take input() if you like
        user_goal = "Write a report on generative AI"

        # IMPORTANT: same thread_id = same memory
        result_state = app.invoke(
            {
                "goal": user_goal,
                "plan": "",
                "result": "",
                "reflection": "",
                "messages": [],
            },
            config={"configurable": {"thread_id": "planner-executor-thread"}},
        )

        print("\n--- PLAN ---")
        print(result_state["plan"])
        print("\n--- RESULT (EXECUTOR) ---")
        print(result_state["result"])
        print("\n--- REFLECTION ---")
        print(result_state["reflection"])
