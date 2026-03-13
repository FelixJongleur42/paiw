#!/usr/bin/env bash
# ============================================================
# paiw container entrypoint
# Starts Ollama in the background, then executes CMD.
# ============================================================
set -euo pipefail

echo "======================================================"
echo "  Pitz' AI Workshop (paiw)"
echo "======================================================"

# ── Ollama ───────────────────────────────────────────────────
echo "[paiw] Starting Ollama server on port 11434 ..."
ollama serve &
OLLAMA_PID=$!

# Wait until Ollama is responsive (max ~30 s)
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/version > /dev/null 2>&1; then
        echo "[paiw] Ollama is ready."
        break
    fi
    sleep 1
done

# ── GPU info ─────────────────────────────────────────────────
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "[paiw] GPU info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version \
               --format=csv,noheader 2>/dev/null || true
else
    echo "[paiw] No NVIDIA GPU detected — running in CPU-only mode."
fi

echo ""
echo "[paiw] Python:  $(python --version)"
echo "[paiw] Node.js: $(node --version)"
echo "[paiw] npm:     $(npm --version)"
echo "[paiw] Ollama:  $(ollama --version 2>/dev/null || echo 'unknown')"
echo ""
echo "[paiw] JupyterLab → http://localhost:8888"
echo "[paiw] Ollama API → http://localhost:11434"
echo "[paiw] Examples   → /workspace/examples"
echo "======================================================"
echo ""

# ── code-server (browser VS Code) ───────────────────────────
if command -v code-server &> /dev/null; then
    echo "[paiw] Starting code-server on port 8080 ..."
    CODE_SERVER_ARGS="--bind-addr $CODE_SERVER_BIND_ADDR"
    if [ -n "${CODE_SERVER_PASSWORD:-}" ]; then
        echo "[paiw] code-server auth: password"
        CODE_SERVER_ARGS="$CODE_SERVER_ARGS --auth password --password $CODE_SERVER_PASSWORD"
    else
        echo "[paiw] code-server auth: none (beware of open port!)"
        CODE_SERVER_ARGS="$CODE_SERVER_ARGS --auth none"
    fi
    # run in background so the main CMD can still start
    code-server $CODE_SERVER_ARGS &
    echo "[paiw] code-server launched."
fi

# ── Execute CMD (default: JupyterLab) ────────────────────────
exec "$@"
