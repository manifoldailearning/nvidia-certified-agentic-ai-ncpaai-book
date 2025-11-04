"""
Advanced Mini Lab:
Two-Agent LangGraph workflow where
1) search_agent pulls context from Wikipedia
2) summary_agent uses Google Gemini to summarize
Requires:
- pip install wikipedia python-dotenv \
    langgraph langchain-core langchain-google-genai

    
- GOOGLE_API_KEY in a .env file
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from wikipedia import summary as wiki_summary
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------------------------------------
# 1. Load env + model
# ----------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

# ----------------------------------------------------
# 2. State schema
# ----------------------------------------------------
class WorkflowState(TypedDict, total=False):
    query: str
    context: Annotated[str, "raw text retrieved from wikipedia"]
    summary: Annotated[str, "LLM-generated summary"]


checkpointer = MemorySaver()

# ----------------------------------------------------
# 3. Agent 1 – retrieve from Wikipedia
# ----------------------------------------------------
def search_agent(state: WorkflowState) -> WorkflowState:
    query = (state.get("query") or "").strip()
    print("\n=== Agent 1: search_agent ===")
    print(f"Query received: {query}")

    if not query:
        state["context"] = "No query provided."
        return state

    try:
        # get first 8–10 sentences to keep tokens low
        ctx = wiki_summary(query, sentences=8, auto_suggest=False, redirect=True)
        state["context"] = ctx
        print("Retrieved content from Wikipedia.")
    except Exception as e:
        # fall back gracefully
        fallback = f"Could not retrieve from Wikipedia for '{query}'. Error: {e}"
        state["context"] = fallback
        print(fallback)

    return state

# ----------------------------------------------------
# 4. Agent 2 – summarize with Gemini
# ----------------------------------------------------
def summary_agent(state: WorkflowState) -> WorkflowState:
    print("\n=== Agent 2: summary_agent ===")
    context = state.get("context") or ""

    if not context:
        state["summary"] = "No context found to summarize."
        return state

    prompt = PromptTemplate.from_template(
        """You are a helpful assistant that summarizes technical/contextual text.
Summarize the following text in 4–6 lines, keep NVIDIA / product names intact:

{context}

Summary:"""
    )

    # chain style: prompt → model
    chain = prompt | gemini
    result = chain.invoke({"context": context})
    state["summary"] = result.content
    print("Summary generated.")
    return state

# ----------------------------------------------------
# 5. Build & compile LangGraph workflow
# ----------------------------------------------------
def build_workflow():
    graph = StateGraph(WorkflowState)
    graph.add_node("search_agent", search_agent)
    graph.add_node("summary_agent", summary_agent)

    graph.add_edge(START, "search_agent")
    graph.add_edge("search_agent", "summary_agent")
    graph.add_edge("summary_agent", END)

    # memory saver so you can replay / inspect
    return graph.compile(checkpointer=checkpointer)

# ----------------------------------------------------
# 6. Run example
# ----------------------------------------------------
if __name__ == "__main__":
    workflow = build_workflow()

    initial_state = {
        "query": "NVIDIA",
    }

    final_state = workflow.invoke(
        initial_state,
        config={"configurable": {"thread_id": "user_1"}}
    )

    print("\n=== Final Output ===")
    print("User query:", initial_state["query"])
    print("\nRetrieved context from wikipedia:")
    print(final_state.get("context"))
    print("\nAgent summary:")
    print(final_state.get("summary"))
