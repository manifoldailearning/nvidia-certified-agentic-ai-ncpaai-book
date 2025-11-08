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
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

for t in [0.2, 0.5, 0.8]:
    llm.temperature = t
    response = llm.invoke(prompt.format(concept="Agent Evaluation"))
    print(f"Temperature {t}: {response.content}\n")
