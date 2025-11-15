# Triton Inference Server Minilab

This minilab demonstrates how to deploy and serve a PyTorch model using NVIDIA Triton Inference Server. The lab includes exporting a simple neural network model to ONNX format and serving it via Triton.

## Overview

This minilab covers:
- Exporting a PyTorch model to ONNX format
- Setting up a Triton model repository
- Running Triton Inference Server in Docker
- Making inference requests via HTTP API

## Prerequisites

- Docker installed and running
- Python 3.11+ (for running the Python scripts)
- Required Python packages: `torch`, `numpy`, `requests`

## Step-by-Step Instructions

### Step 1 — Export the Model (Optional)

If you haven't already exported the model, you can run the export script:

```bash
python export_model.py
```

This will create `model.onnx` and `model.onnx.data` in the current directory. The model repository already contains the exported model, so this step is optional.

### Step 2 — Verify Model Repository Structure

Ensure your model repository structure is correct:

```
models/
└── nemo_model/
    ├── config.pbtxt
    └── 1/
        ├── model.onnx
        └── model.onnx.data
```

The model repository is already set up in the `models/` directory.

### Step 3 — Pull the Triton Server Docker Image

Pull the NVIDIA Triton Inference Server Docker image:

```bash
docker pull nvcr.io/nvidia/tritonserver:25.10-py3
```

### Step 4 — Run the Triton Server

Start the Triton Inference Server with the model repository:

```bash
docker run --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:25.10-py3 \
  tritonserver --model-repository=/models
```

**Note:** Make sure you run this command from the `minilab_triton` directory so that the `models` directory path is correct.

You should see output similar to:

```
loading: nemo_model:1
successfully loaded 'nemo_model' version 1
HTTP available at :8000
gRPC available at :8001
Metrics at :8002
```

The server is now running and ready to accept inference requests.

### Step 5 — Verify the Model Is Loaded

#### Health Check

Check if the server is ready:

```bash
curl http://localhost:8000/v2/health/ready
```

You should receive a response indicating the server is ready.

#### Inspect the Model

Get detailed information about the loaded model:

```bash
curl http://localhost:8000/v2/models/nemo_model
```

Expected JSON response:

```json
{
  "name": "nemo_model",
  "versions": ["1"],
  "platform": "onnxruntime_onnx",
  "inputs": [{"name": "INPUT__0", "datatype": "FP32", "shape": [-1, 4]}],
  "outputs": [{"name": "OUTPUT__0", "datatype": "FP32", "shape": [-1, 3]}]
}
```

### Step 6 — Run Inference Requests

#### Method 1: Using cURL (Simplest)

Make an inference request using cURL:

```bash
curl -X POST http://localhost:8000/v2/models/nemo_model/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "INPUT__0",
        "shape": [1, 4],
        "datatype": "FP32",
        "data": [1.0, 2.0, 3.0, 4.0]
      }
    ]
  }'
```

Example output:

```json
{
  "model_name": "nemo_model",
  "model_version": "1",
  "outputs": [
    {
      "name": "OUTPUT__0",
      "datatype": "FP32",
      "shape": [1,3],
      "data": [-2.1480, 1.2838, -0.8980]
    }
  ]
}
```

#### Method 2: Using Python Client

Run the Python inference client:

```bash
python infer_client.py
```

This script will:
- Generate a random input vector (1 x 4)
- Send an inference request to the Triton server
- Print the response with formatted JSON

## Model Details

- **Model Type:** Simple Linear Neural Network
- **Input:** 4-dimensional vector (FP32)
- **Output:** 3-dimensional vector (FP32)
- **Platform:** ONNX Runtime

## Troubleshooting

### Port Already in Use

If you get an error about ports being in use, you can:
- Stop any existing Triton server containers
- Change the port mappings (e.g., `-p 9000:8000`)

### Model Not Loading

- Verify the model repository path is correct
- Check that `config.pbtxt` is properly formatted
- Ensure the ONNX model files are in the `1/` directory
- Check Docker logs for detailed error messages

### Connection Refused

- Ensure the Triton server container is running
- Verify the ports are correctly mapped
- Check that the server has finished loading models (wait for "successfully loaded" message)

## Additional Resources

- [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Triton Inference Server GitHub](https://github.com/triton-inference-server/server)

