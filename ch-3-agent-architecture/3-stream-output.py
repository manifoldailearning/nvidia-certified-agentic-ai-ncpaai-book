from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict, List

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

class ConversationState(TypedDict):
    messages: List[str]

graph = StateGraph(ConversationState)

def talk(state: ConversationState) -> ConversationState:
    user_messages = state["messages"]

    # stream from LLM and print live
    for chunk in llm.stream(user_messages[-1]):
        print(chunk.content, end="", flush=True)

    print()  # newline after stream
    # you can append the last user msg to history
    return state

# 3. build graph
graph = StateGraph(ConversationState)
graph.add_node("talk", talk)
graph.add_edge(START, "talk")
graph.add_edge("talk", END)
app = graph.compile()

# 4. IMPORTANT: stream the app, don't just invoke it
for _ in app.stream({"messages": ["Explain token streaming in one line."]}):
    # we already printed inside the node, so we can ignore the event
    pass