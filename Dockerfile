# ============================================================
# Pitz' AI Workshop (paiw) — Docker Image
# ============================================================
# Base: Ubuntu 22.04 with NVIDIA CUDA 12.3 + cuDNN 9
#
# Includes:
#   - Python 3.11 (deadsnakes PPA)
#   - Node.js 20 LTS
#   - Claude CLI (@anthropic-ai/claude-code)
#   - Ollama (local LLM runner)
#   - PyTorch 2.x (CUDA 12.1 wheels)
#   - HuggingFace Transformers, Accelerate, PEFT, Datasets
#   - Sentence-Transformers (embeddings)
#   - LangChain + LlamaIndex (RAG frameworks)
#   - ChromaDB, FAISS, Qdrant (vector stores)
#   - Anthropic & OpenAI Python clients
#   - JupyterLab
#
# Platform notes:
#   - WSL2 (Windows x64) with NVIDIA GPU: full CUDA acceleration
#   - Linux x64 with NVIDIA GPU:           full CUDA acceleration
#   - Linux / macOS without NVIDIA GPU:    CPU-only (no CUDA accel)
#   - Apple Silicon macOS:                 CPU via Docker Desktop VM
#   GPU passthrough requires the NVIDIA Container Toolkit on the host.
# ============================================================

FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# ── Environment ──────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build arguments
ARG PYTHON_VERSION=3.11
ARG NODE_MAJOR=20

WORKDIR /workspace

# ── System packages ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        # Essentials
        build-essential \
        cmake \
        curl \
        wget \
        git \
        vim \
        nano \
        htop \
        tmux \
        jq \
        unzip \
        zip \
        ca-certificates \
        gnupg \
        lsb-release \
        software-properties-common \
        # Compression utilities needed by external installers
        zstd \
        # SSL / FFI (needed by some pip packages)
        libssl-dev \
        libffi-dev \
        # GUI / rendering libs required by some Python packages
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        # Document processing
        poppler-utils \
        tesseract-ocr \
        # Audio / video (multimodal use-cases)
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Python 3.11 (deadsnakes PPA) ─────────────────────────────
RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap pip for the new Python and set it as the system default
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 10 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 10 \
    && update-alternatives --install /usr/bin/pip     pip     /usr/local/bin/pip${PYTHON_VERSION} 10 \
    && pip install --upgrade pip setuptools wheel

# ── Node.js 20 LTS ───────────────────────────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# ── Claude CLI ───────────────────────────────────────────────
# Anthropic's Claude Code CLI — requires ANTHROPIC_API_KEY at runtime
RUN npm install -g @anthropic-ai/claude-code

# ── Ollama ───────────────────────────────────────────────────
# Installs the `ollama` binary to /usr/local/bin.
# The server is started by entrypoint.sh; models live in /root/.ollama
# (mount a volume there to persist downloaded models across container restarts).
RUN curl -fsSL https://ollama.ai/install.sh | sh

# ── PyTorch 2.x (CUDA 12.1 wheels) ──────────────────────────
# cu121 wheels are compatible with the CUDA 12.3 runtime in this image.
RUN pip install \
        torch==2.2.2 \
        torchvision==0.17.2 \
        torchaudio==2.2.2 \
        --index-url https://download.pytorch.org/whl/cu121

# ── Core ML / LLM libraries ──────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# add a handy notebook/console helper for executing code cells from the
# lab UI (Run ▶️ buttons).
RUN pip install jupyterlab-code-runner

# ── Jupyter LSP & language servers ───────────────────────────
# install the JupyterLab language-server extension plus a handful of
# common language servers referenced by the startup logs.  Mostly
# optional; the container works fine without these, but the warning
# message is less noisy when they're present.
RUN pip install jupyterlab-lsp python-lsp-server[all] \
    && npm install -g bash-language-server \
                   dockerfile-language-server-nodejs \
                   vscode-css-languageserver-bin \
                   vscode-html-languageserver-bin \
                   vscode-json-languageserver-bin \
                   yaml-language-server

# ── code-server (VS Code in the browser) ──────────────────────
# We install the official code-server release and configure sensible
# defaults via environment variables.  The container will expose
# port 8080 and the entrypoint script starts the service if present.
# Password protection may be enabled at runtime by setting
# CODE_SERVER_PASSWORD in the compose file or .env.
RUN curl -fsSL https://code-server.dev/install.sh | sh

# expose extra port for web-based VS Code
EXPOSE 8080

# default code-server environment
ENV CODE_SERVER_BIND_ADDR=0.0.0.0:8080 \
    CODE_SERVER_AUTH=none \
    CODE_SERVER_PASSWORD=${CODE_SERVER_PASSWORD:-}

# ── Workspace structure ──────────────────────────────────────
RUN mkdir -p /workspace/examples \
             /workspace/data \
             /workspace/models \
             /workspace/notebooks

COPY examples/ /workspace/examples/

# ── Ports ────────────────────────────────────────────────────
# 8888  — JupyterLab
# 11434 — Ollama REST API
# 8000  — FastAPI / vLLM / custom app
EXPOSE 8888 11434 8000

# ── Entrypoint ───────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--notebook-dir=/workspace"]
