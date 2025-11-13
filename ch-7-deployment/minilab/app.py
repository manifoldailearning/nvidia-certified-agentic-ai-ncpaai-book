# app.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize FastAPI
app = FastAPI(title="Gemini Agentic API")

# Initialize Gemini model (fast + cheap + great for teaching)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # or "gemini-2.0-flash"
    api_key=GOOGLE_API_KEY
)

# Request/response schema
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    response: str

@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    """
    Simple agent-style endpoint:
    - Accepts a natural language question
    - Sends it to Gemini
    - Returns the generated response
    """
    result = llm.invoke(payload.question)
    return AskResponse(response=result.content)
