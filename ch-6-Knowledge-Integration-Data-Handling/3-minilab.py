"""
Conversational RAG with Memory (fixed)
- keeps history across turns
- uses SQLite checkpointer
"""

from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# ----- 1. state -----
class ChatState(TypedDict):
    user_input: str
    chat_history: list[str]
    answer: str

# ----- 2. vector store -----
docs = [
    "FAISS is a library for efficient similarity search on embeddings.",
    "Vector databases store embedding vectors for semantic search.",
    "RAG (Retrieval-Augmented Generation) combines retrieval with LLM generation.",
    "Persistent memory helps an AI agent recall previous user interactions."
]
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = FAISS.from_texts(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(k=2)

# ----- 3. llm -----
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

# ----- 4. node -----
def rag_with_memory(state: ChatState) -> ChatState:
    """Retrieve from FAISS, mix with prior turns, answer with Gemini."""
    query = state["user_input"]
    history_text = "\n".join(state.get("chat_history", []))

    relevant = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in relevant)

    prompt = (
        "You are a conversational AI assistant. Use BOTH the conversation history "
        "and the retrieved context to answer.\n"
        "If the reference is vague (e.g. 'that'), use the last user turn from history.\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"User: {query}\nAssistant:"
    )

    resp = llm.invoke(prompt)

    # update history
    history = state.get("chat_history", [])
    history.append(f"User: {query}")
    history.append(f"AI: {resp.content}")

    state["chat_history"] = history
    state["answer"] = resp.content
    return state

# ----- 5. graph -----
workflow = StateGraph(ChatState)
workflow.add_node("rag_memory", rag_with_memory)
workflow.add_edge(START, "rag_memory")
workflow.add_edge("rag_memory", END)

if __name__ == "__main__":
    # use sqlite properly
    with SqliteSaver.from_conn_string("rag_memory.db") as checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
        THREAD_ID = "rag-memory-session-1"

        print("Type 'exit' to stop.\n")
        while True:
            user_query = input("You: ").strip()
            if user_query.lower() in ("exit", "quit"):
                break

            # IMPORTANT: only send the new user input
            # let the checkpointer restore chat_history
            state = {"user_input": user_query}

            result = app.invoke(
                state,
                config={"configurable": {"thread_id": THREAD_ID}},
            )
            print("AI:", result["answer"], "\n")
