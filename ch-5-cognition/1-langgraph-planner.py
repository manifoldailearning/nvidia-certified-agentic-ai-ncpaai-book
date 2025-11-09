from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

# Define the agent state
class AgentState(TypedDict):
    goal: str
    plan: str
    result: str
# ---------- 2. Planner node ----------
def planner(state: AgentState) -> AgentState:
    goal = state["goal"]
    plan = (
        f"Goal: {goal}\n"
        "Plan:\n"
        "1) Identify 3 major themes related to the goal.\n"
        "2) Provide short, factual explanations for each.\n"
        "3) Summarize insights in concise bullet points."
    )
    state["plan"] = plan
    return state

# ---------- 3. Executor node ----------
def executor(state: AgentState) -> AgentState:
    goal = state["goal"]
    plan = state["plan"]

    prompt = (
        "You are an AI researcher assistant.\n"
        f"User goal: {goal}\n"
        f"{plan}\n\n"
        "Now execute the plan and provide your response.\n"
        "- Use bullet points.\n"
        "- Keep each explanation 1–2 lines.\n"
        "- Focus on clarity, not technical depth."
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    state["result"] = response.content
    return state

# ---------- 4. Build LangGraph ----------
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner)
workflow.add_node("executor", executor)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", END)

app = workflow.compile()

# ---------- 5. Run ----------
if __name__ == "__main__":
    output = app.invoke({"goal": "Summarize key trends in AI research"})
    print(output["result"])