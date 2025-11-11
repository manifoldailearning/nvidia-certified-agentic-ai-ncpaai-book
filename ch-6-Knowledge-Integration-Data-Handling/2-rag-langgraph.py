"""
RAG with LangGraph

This script demonstrates Retrieval-Augmented Generation (RAG) using:
1. FAISS vector store for document retrieval
2. Google Generative AI embeddings
3. LangGraph workflow for orchestration
4. Gemini 2.5 Flash as the LLM

Requires:
- pip install langchain-community langchain-google-genai langgraph langchain-text-splitters python-dotenv
- GOOGLE_API_KEY in a .env file
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# ----------------------------------------------------
# 1. Load environment variables
# ----------------------------------------------------
load_dotenv()

# ----------------------------------------------------
# 2. Build FAISS vector store for retrieval
# ----------------------------------------------------
docs = [
    "A vector database stores embedding vectors for fast similarity search.",
    "FAISS is a local vector store useful for prototyping.",
    "PGVector is a Postgres extension for enterprise vector search.",
    "Vector databases power Retrieval-Augmented Generation (RAG)."
]

splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=0)
chunks = splitter.split_text(" ".join(docs))

# Use Google Generative AI embeddings (can be swapped for other providers)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

retriever = vectorstore.as_retriever(k=3)

# ----------------------------------------------------
# 3. Initialize Gemini 2.5 Flash as the LLM
# ----------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

# ----------------------------------------------------
# 4. Define state schema
# ----------------------------------------------------
class RAGState(TypedDict):
    query: str
    answer: str

# ----------------------------------------------------
# 5. Define RAG node
# ----------------------------------------------------
def rag_node(state: RAGState) -> RAGState:
    """Retrieve context from FAISS and answer with Gemini."""
    query = state["query"]

    # Newer retrievers use .invoke(query)
    try:
        relevant_docs = retriever.invoke(query)
    except AttributeError:
        # Fallback for older versions
        relevant_docs = retriever._get_relevant_documents(query)

    context = "\n\n".join(d.page_content for d in relevant_docs)

    prompt = (
        "You are a helpful AI assistant. Use ONLY the context to answer.\n"
        "If the answer is not in the context, say you don't have enough data.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    resp = llm.invoke(prompt)
    state["answer"] = resp.content
    return state

# ----------------------------------------------------
# 6. Build LangGraph workflow
# ----------------------------------------------------
workflow = StateGraph(RAGState)
workflow.add_node("rag", rag_node)
workflow.add_edge(START, "rag")
workflow.add_edge("rag", END)
app = workflow.compile()

# ----------------------------------------------------
# 7. Run example
# ----------------------------------------------------
if __name__ == "__main__":
    init_state: RAGState = {
        "query": "Explain vector databases",
        "answer": "",
    }
    result = app.invoke(init_state)
    print("\n--- RAG OUTPUT ---")
    print(result["answer"])