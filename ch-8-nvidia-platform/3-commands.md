```bash
docker login nvcr.io
docker pull nvcr.io/nvidia/pytorch:23.10-py3
ngc registry model download-version nvidia/nemo:2.0
ngc registry model list --framework pytorch
```
