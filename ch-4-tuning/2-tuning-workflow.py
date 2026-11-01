# Prompt-tuning experiment
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

prompt = ChatPromptTemplate.from_template("Explain {concept} in simple 2 sentences.")

# LLM for summarization
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=GOOGLE_API_KEY,
)

for level in ["low", "medium", "high"]:
    llm.reasoning_effort = level   # Gemini "thinking level"; temperature is ignored on Gemini 3.6+
    response = llm.invoke(prompt.format(concept="Agent Evaluation"))
    print(f"Thinking level {level}: {response.text}\n")
