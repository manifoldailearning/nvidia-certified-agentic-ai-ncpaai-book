# NVIDIA NIM hosted API (OpenAI-compatible). Requires NVIDIA_API_KEY from build.nvidia.com.
import os, requests

response = requests.post(
    "https://integrate.api.nvidia.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}"},
    json={
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [{"role": "user", "content": "Explain RAG optimization techniques."}],
    },
)
print(response.json()["choices"][0]["message"]["content"])
