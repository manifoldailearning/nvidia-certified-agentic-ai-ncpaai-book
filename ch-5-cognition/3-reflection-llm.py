from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()  # ensure GOOGLE_API_KEY is set in .env

# ---- Initialize model ----
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# ---- Stage 1: Generate initial response ----
prompt = """Summarize the key trends in Artificial Intelligence for 2025. 
            Display at max 5 bullet points"""
draft = llm.invoke([HumanMessage(content=prompt)])
print("\n--- FIRST OUTPUT ---")
print(draft.text)

# ---- Stage 2: Reflect on the generated output ----
reflection_prompt = (
    f"Evaluate the following answer for completeness, clarity, and factual accuracy.\n\n"
    f"Answer:\n{draft.text}\n\n"
    f"Provide constructive feedback in 2–3 sentences."
)
reflection = llm.invoke([HumanMessage(content=reflection_prompt)])
print("\n--- REFLECTION ---")
print(reflection.text)

# ---- Stage 3: Improve based on reflection ----
improvement_prompt = (
    f"Rewrite the original answer incorporating the feedback below.\n\n"
    f"Original Answer:\n{draft.text}\n\n"
    f"Feedback:\n{reflection.text}\n\n"
    f"Produce the improved final version."
)
improved = llm.invoke([HumanMessage(content=improvement_prompt)])
print("\n--- IMPROVED OUTPUT ---")
print(improved.text)
