# LangSmith evaluation (requires LANGSMITH_API_KEY and a dataset named "agentic-ai-eval"
# with inputs {"question": ...} and reference outputs {"answer": ...}).
import os
from dotenv import load_dotenv
from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

def agent_response(question: str) -> str:
    """Stand-in for your agent: here, a single Gemini call."""
    return llm.invoke(question).text

client = Client()

def production_agent(inputs: dict) -> dict:
    return {"answer": agent_response(inputs["question"])}   # your agent

def accuracy(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    return reference_outputs["answer"].lower() in outputs["answer"].lower()

results = client.evaluate(
    production_agent,
    data="agentic-ai-eval",
    evaluators=[accuracy],          # add coherence/safety evaluators the same way
    experiment_prefix="production-agent",
)
print(results.to_pandas())
