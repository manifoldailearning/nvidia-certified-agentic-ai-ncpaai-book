# Prompt-tuning experiment
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

prompt = PromptTemplate.from_template("Explain {concept} in simple terms.")

# LLM for summarization
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

for t in [0.2, 0.5, 0.8]:
    llm.temperature = t
    response = llm.invoke(prompt.format(concept="Agent Evaluation"))
    print(f"Temperature {t}: {response}")
