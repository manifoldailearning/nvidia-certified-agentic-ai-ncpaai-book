# snippet from the book - for reference only
import requests

response = requests.post(
    "https://api.nvidia.com/v1/llm",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"prompt": "Explain RAG optimization techniques."}
)
print(response.json())
