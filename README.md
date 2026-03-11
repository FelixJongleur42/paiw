# paiw — Pitz' AI Workshop

A ready-to-use Docker environment for experimenting with AI: local LLMs via
Ollama, RAG pipelines, embeddings, inference with HuggingFace Transformers, and
the Claude / OpenAI APIs — all in one container.

## What's inside

| Component | Details |
|-----------|---------|
| **Base** | Ubuntu 22.04 + NVIDIA CUDA 12.3 / cuDNN 9 |
| **Python** | 3.11 (deadsnakes PPA) |
| **Node.js** | 20 LTS |
| **Claude CLI** | `@anthropic-ai/claude-code` (requires `ANTHROPIC_API_KEY`) |
| **Ollama** | Local LLM runner (Llama 3, Mistral, Gemma, Phi, …) |
| **PyTorch** | 2.2 with CUDA 12.1 wheels |
| **Transformers** | HuggingFace Transformers, Accelerate, PEFT, Datasets |
| **Embeddings** | sentence-transformers, FlagEmbedding |
| **RAG** | LangChain, LlamaIndex |
| **Vector stores** | ChromaDB, FAISS, Qdrant |
| **API clients** | `anthropic`, `openai` |
| **Jupyter** | JupyterLab 4 |

## Platform support

| Platform | GPU acceleration |
|----------|-----------------|
| WSL2 (Windows x64) with NVIDIA GPU | ✅ CUDA (requires NVIDIA Container Toolkit) |
| Linux x64 with NVIDIA GPU | ✅ CUDA (requires NVIDIA Container Toolkit) |
| Linux / Windows without NVIDIA GPU | CPU only |
| macOS (Intel or Apple Silicon) | CPU only (via Docker Desktop) |

## Quick start

### 1 — Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Compose v2
- *(GPU)* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- *(WSL2 GPU)* [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

### 2 — Configure API keys

```bash
cp .env.example .env
# Open .env and fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, HUGGINGFACE_TOKEN
```

### 3 — Build and run (with GPU)

```bash
docker compose up --build
```

### 3 — Build and run (CPU only — no NVIDIA GPU)

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

### 4 — Open JupyterLab

Navigate to **http://localhost:8888** in your browser.

### 5 — Pull an Ollama model

```bash
# From the JupyterLab terminal or any terminal inside the container:
ollama pull llama3.2:3b
```

## Services

| Port | Service |
|------|---------|
| 8888 | JupyterLab |
| 11434 | Ollama REST API |
| 8000 | Your FastAPI / custom app |

## Persisting Ollama models

Downloaded models are stored in the `ollama_models` Docker volume defined in
`docker-compose.yml`. They survive container restarts automatically.

To mount a host directory instead, change the volume mapping in
`docker-compose.yml`:

```yaml
volumes:
  - /path/on/host/.ollama:/root/.ollama
```

## Examples

See [`examples/README.md`](examples/README.md) for a description of all
included example scripts.

```
examples/
├── 01_basic_llm_inference.py   # HuggingFace Transformers text generation
├── 02_ollama_chat.py           # Interactive chat with a local Ollama model
├── 03_embeddings.py            # Sentence embeddings + FAISS similarity search
├── 04_rag_basic.py             # RAG with LangChain + ChromaDB + Ollama
└── 05_claude_example.py        # Claude API: single-turn, streaming, multi-turn
```

## Using the Claude CLI

```bash
# Inside the container terminal:
export ANTHROPIC_API_KEY=sk-ant-...
claude
```

## Directory layout inside the container

```
/workspace/
├── examples/   ← mounted from ./examples on the host
├── data/       ← mounted from ./data on the host (put your documents here)
├── models/     ← for locally saved model checkpoints
└── notebooks/  ← create your own Jupyter notebooks here
```
