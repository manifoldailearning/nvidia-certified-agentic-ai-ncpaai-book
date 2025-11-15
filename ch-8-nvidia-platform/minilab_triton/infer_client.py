import json
import numpy as np
import requests

# Triton HTTP inference URL
url = "http://localhost:8000/v2/models/nemo_model/infer"

# Create a random input vector (1 x 4)
data = np.random.randn(1, 4).astype("float32").tolist()

payload = {
    "inputs": [
        {
            "name": "INPUT__0",
            "shape": [1, 4],
            "datatype": "FP32",
            "data": data
        }
    ]
}

response = requests.post(url, json=payload)
print("Status code:", response.status_code)
print("Response JSON:")
print(json.dumps(response.json(), indent=2))
