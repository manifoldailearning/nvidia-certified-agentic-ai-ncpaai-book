```bash
docker run --gpus all \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  nvcr.io/nvidia/tritonserver:23.08-py3 tritonserver \
  --model-repository=/models
```