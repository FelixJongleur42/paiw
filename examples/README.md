# Examples — Pitz' AI Workshop

Each script in this folder is a self-contained, runnable example.
Start them from the JupyterLab terminal (`/workspace/examples`) or via the
file browser.

## Prerequisites

Copy `.env.example` to `.env` at the repository root and fill in any API keys
you want to use before starting the container.

## Examples overview

| File | What it demonstrates |
|------|----------------------|
| `01_basic_llm_inference.py` | Text generation with HuggingFace Transformers (CPU / GPU) |
| `02_ollama_chat.py` | Interactive chat with a locally running Ollama model |
| `03_embeddings.py` | Sentence embeddings with `sentence-transformers`; similarity search |
| `04_rag_basic.py` | Retrieval-Augmented Generation (RAG) using LangChain + ChromaDB |
| `05_claude_example.py` | Conversational exchange with Anthropic's Claude API |

## Running an example

```bash
# From the JupyterLab terminal:
cd /workspace/examples
python 01_basic_llm_inference.py
```

## Pulling an Ollama model

```bash
# Pull a small but capable model (requires internet access from the container)
ollama pull llama3.2:3b

# List available local models
ollama list
```

## Useful ports

| Port  | Service |
|-------|---------|
| 8888  | JupyterLab |
| 11434 | Ollama REST API |
| 8000  | Your FastAPI / custom app |
