# 4.6 Mini Lab – Evaluate and Tune an Agent 
# Runs locally with simple evaluation — no LangSmith needed.
# Add your Google API key to a .env file as GOOGLE_API_KEY

from dotenv import load_dotenv
import os
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

# 1. Agent logic
def agent_response(prompt):
    """Simple wrapper to invoke the Gemini model."""
    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)

# 2. Simple evaluation metrics
def evaluate_agent(agent_fn, prompts, answers):
    """Evaluates accuracy and coherence heuristically."""
    results = []
    for prompt, expected in zip(prompts, answers):
        output = agent_fn(prompt)
        accuracy = 1.0 if expected.lower() in output.lower() else 0.5
        coherence = 1.0 if len(output.split()) > 5 else 0.5
        results.append({"prompt": prompt, "accuracy": accuracy, "coherence": coherence})
    return results

# 3. Sample dataset
dataset = [
    ("What is the capital of France?", "Paris"),
    ("Who wrote Hamlet?", "Shakespeare"),
    ("What is 2 + 2?", "4"),
]

# 4. Evaluate agent performance
scores = evaluate_agent(agent_response, [q for q, _ in dataset], [a for _, a in dataset])

# 5. Display results
for s in scores:
    print(f"Prompt: {s['prompt']}")
    print(f"  Accuracy: {s['accuracy']:.2f}, Coherence: {s['coherence']:.2f}\n")

# 6. Aggregate metrics
avg_accuracy = np.mean([s["accuracy"] for s in scores])
avg_coherence = np.mean([s["coherence"] for s in scores])
print(f"Average Accuracy: {avg_accuracy:.2f}, Average Coherence: {avg_coherence:.2f}")
