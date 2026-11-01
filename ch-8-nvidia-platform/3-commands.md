```bash
docker login nvcr.io
docker pull nvcr.io/nvidia/pytorch:26.10-py3
ngc registry model download-version <org>/[<team>/]<model_name>:<version>   # e.g. a Nemotron checkpoint from the NGC catalog
ngc registry model list "nvidia/*"
```
