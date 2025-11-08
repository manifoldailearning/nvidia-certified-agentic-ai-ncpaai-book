import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

# 2. Original prediction (with an intentional error)
context = "Agent predicted: The capital of Canada is Toronto."

# 3. Reflection prompt
reflection_prompt = (
    "You previously gave an answer. "
    "Review it for factual accuracy and explain if it needs correction. "
    "Provide a refined, correct statement."
)

# 4. Reflect and correct
review = llm.invoke(reflection_prompt + "\n\n" + context)
print(review.content)
