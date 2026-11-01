# NCP-AAI Exam Prep Guide: Companion Code

Companion code for the book **"NVIDIA Certified Agentic AI Professional (NCP-AAI): Unofficial Exam Prep Guide"** by Nachiketh Murthy (Manifold AI Learning).

Each folder holds the scripts, configs and mini labs from one chapter of the book. The Python examples use LangChain and LangGraph with Google Gemini, and the NVIDIA platform chapters include Dynamo-Triton (Triton), NIM and NeMo Guardrails examples.

> **Unofficial.** This book and repository are independent study material. They are not affiliated with, endorsed by, or sponsored by NVIDIA. NVIDIA, NeMo, NIM, Triton and Dynamo are trademarks of NVIDIA Corporation.

## Setup

You need Python 3.11 or newer.

**Option A: venv**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Option B: conda**

```bash
conda create --prefix ./my_env python=3.11 -y
conda activate ./my_env
pip install -r requirements.txt
```

**API key.** Create a Gemini API key at https://aistudio.google.com/api-keys and put it in a `.env` file in the folder you run the scripts from:

```bash
GOOGLE_API_KEY=your-key-here
```

The examples use `gemini-3.8-flash`. New API keys cannot use the Gemini 2.5 models that the first edition used.

Some labs need extra keys or tools:

- `ch-3-agent-architecture/5-minilab.py`: an Alpha Vantage key (free at https://www.alphavantage.co/support/#api-key). The lab falls back to sample data without one.
- `ch-4-tuning/1-langsmith-eval.py`: `LANGSMITH_API_KEY` and a LangSmith dataset named `agentic-ai-eval`.
- `ch-8-nvidia-platform/5-sample-api-call.py`: `NVIDIA_API_KEY` from https://build.nvidia.com.
- `ch-7-deployment` and `ch-8-nvidia-platform/minilab_triton`: Docker (and an NVIDIA GPU for `--gpus all`).

Run a script from its own folder, for example:

```bash
cd ch-3-agent-architecture
python 1-react-agent.py
```

## Folder-to-chapter map

| Folder | Book chapter |
|---|---|
| `ch-2-agent-development/` | Ch.2 Agent Architecture and Design (two-agent search and summary lab) |
| `ch-3-agent-architecture/` | Ch.3 Agent Development (ReAct loop, retries, streaming, reflection, tool-using mini lab) |
| `ch-4-tuning/` | Ch.4 Evaluation and Tuning (LangSmith evaluation, thinking-level experiment, eval mini lab) |
| `ch-5-cognition/` | Ch.5 Cognition, Planning and Memory (planner/executor, SQLite memory, reflection) |
| `ch-6-Knowledge-Integration-Data-Handling/` | Ch.6 Knowledge Integration and Data Handling (FAISS, RAG with LangGraph, conversational RAG) |
| `ch-7-deployment/` | Ch.7 Deployment and Scaling (Dockerfile, Kubernetes, Triton config, FastAPI mini lab) |
| `ch-8-nvidia-platform/` | Ch.8 NVIDIA Platform Implementation (NeMo, Dynamo-Triton, NIM, NeMo Guardrails, Triton mini lab) |
| `ch-9-run-monitor-maintain/` | Ch.9 Run, Monitor and Maintain (Prometheus metrics, drift detection, logging) |
| `ch-10-safety-ethics/` | Ch.10 Safety, Ethics and Compliance (input/output guards, PII masking, bias check, guardrail graph) |
| `ch-11-human-ai/` | Ch.11 Human-AI Interaction and Oversight (explainability, audit trail, human-in-the-loop graph) |

Files marked "snippet from the book" or "illustrative" are reference snippets, not complete programs.

## 2026 revision notes

- **Model:** all Gemini examples now use `gemini-3.8-flash`. `temperature` is no longer set, because Gemini 3.6 and later ignore it; the Ch.4 tuning experiment now varies the thinking level (`reasoning_effort`) instead.
- **Response text:** Gemini 3 models return a list of content blocks, so the scripts read model output with `.text` instead of `.content`.
- **LangChain 1.x / LangGraph 1.x:** requirements now pin the 1.x releases. Retrievers take `as_retriever(search_kwargs={"k": ...})`, the Ch.4 LangSmith example uses the current `Client.evaluate(target, data=..., evaluators=[...])` API, and embeddings use `models/gemini-embedding-001` (`text-embedding-004` was shut down). `langchain-community` is being sunset; it is still used here only for FAISS.
- **NVIDIA renames:** Triton Inference Server is now **NVIDIA Dynamo-Triton** (the container is still `nvcr.io/nvidia/tritonserver`, now 26.10). The NIM example uses the OpenAI-compatible `https://integrate.api.nvidia.com/v1/chat/completions` endpoint, and the NeMo Guardrails example uses the current `rails.input.flows` / `rails.output.flows` format.
- **NeMo Agent Toolkit:** formerly the Agent Intelligence (AIQ) Toolkit. NVIDIA deprecated it in October 2026 (end of life at the end of 2026) in favor of NeMo Relay, NeMo Helix and NeMo Fabric. The exam study guide still names it, so the book still covers it.
- **ONNX export:** the Triton mini lab needs `onnxscript` with current PyTorch and uses `dynamic_shapes` instead of `dynamic_axes`.
