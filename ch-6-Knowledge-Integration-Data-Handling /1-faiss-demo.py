from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

docs = ["LangChain enables modular AI agents.",
        "FAISS stores embeddings for local retrieval."]

splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
chunks = splitter.split_text(' '.join(docs))

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

query = "What does FAISS do?"
results = vectorstore.similarity_search(query)
print(results[0].page_content)
