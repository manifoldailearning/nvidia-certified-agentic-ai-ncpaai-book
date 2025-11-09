from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
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

class ChatState(TypedDict):
    # automatically merged by LangGraph
    messages: Annotated[list[HumanMessage], add_messages]
    user_input: str

# ---------- 2. Define nodes ----------
def user_node(state: ChatState):
    # return the user message as a list to be merged
    return {"messages": [HumanMessage(content=state["user_input"])]}

def chat_node(state: ChatState):
    # model sees all merged messages so far
    response = llm.invoke(
        state["messages"] + [HumanMessage(content="Respond to the user briefly.")]
    )
    # returning list ensures add_messages merges it automatically
    return {"messages": [HumanMessage(content=response.content)]}

# ---------- 3. Build graph ----------
workflow = StateGraph(ChatState)
workflow.add_node("user", user_node)
workflow.add_node("chat", chat_node)
workflow.add_edge(START, "user")
workflow.add_edge("user", "chat")
workflow.add_edge("chat", END)

# ---------- 4. Run with SQLite checkpointer ----------
if __name__ == "__main__":
    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
        THREAD_ID = "user-1" # modify per user/session

        print("Type 'exit' to stop.\n")
        while True:
            user_text = input("You: ").strip()
            if user_text.lower() in ("exit", "quit"):
                break

            state = {"messages": [], "user_input": user_text}
            result = app.invoke(
                state,
                config={"configurable": {"thread_id": THREAD_ID}},
            )
            # get the model’s latest message (the last in merged list)
            print("AI:", result["messages"][-1].content)