"""
FAISS Vector Store Demo

This script demonstrates how to:
1. Create text chunks using RecursiveCharacterTextSplitter
2. Generate embeddings using Google Generative AI
3. Store embeddings in a FAISS vector store
4. Perform similarity search queries

Requires:
- pip install langchain-community langchain-google-genai langchain-text-splitters python-dotenv
- GOOGLE_API_KEY in a .env file
"""

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# ----------------------------------------------------
# 1. Load environment variables
# ----------------------------------------------------
load_dotenv()

# ----------------------------------------------------
# 2. Prepare documents and create text chunks
# ----------------------------------------------------
docs = ["LangChain enables modular AI agents.",
        "FAISS stores embeddings for local retrieval."]

splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
chunks = splitter.split_text(' '.join(docs))

# ----------------------------------------------------
# 3. Initialize embeddings and create vector store
# ----------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

# ----------------------------------------------------
# 4. Perform similarity search
# ----------------------------------------------------
query = "What does FAISS do?"
results = vectorstore.similarity_search(query)
print(results[0].page_content)
