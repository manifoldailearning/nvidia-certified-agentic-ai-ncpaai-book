Run NVIDIA Dynamo-Triton (formerly Triton Inference Server) with a local model repository.
Run from the directory that contains `models/`:

```bash
docker run --gpus all --rm \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:26.10-py3 tritonserver \
  --model-repository=/models
```
