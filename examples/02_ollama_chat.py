"""
02_ollama_chat.py
─────────────────
Interactive CLI chat with a locally running Ollama model.

Demonstrates:
  - Checking Ollama server availability
  - Streaming responses token-by-token
  - Multi-turn conversation history

Prerequisites:
    # Pull a model first (run inside the container):
    ollama pull llama3.2:3b

Usage:
    python 02_ollama_chat.py
    python 02_ollama_chat.py --model llama3.2:3b
    python 02_ollama_chat.py --model mistral:7b --system "You are a helpful coding assistant."
"""

import argparse
import sys

import ollama


SYSTEM_DEFAULT = (
    "You are a knowledgeable AI assistant. "
    "Answer clearly and concisely."
)


def check_server() -> None:
    """Raise a friendly error if Ollama is not reachable."""
    try:
        models = ollama.list()
        available = [m.model for m in models.models]
        if available:
            print(f"[ollama] Available models: {', '.join(available)}")
        else:
            print("[ollama] No models pulled yet. Run: ollama pull llama3.2:3b")
    except Exception as exc:
        print(f"[error] Cannot reach Ollama server: {exc}")
        print("        Make sure the container was started via entrypoint.sh,")
        print("        or run `ollama serve` in a separate terminal.")
        sys.exit(1)


def chat(model: str, system_prompt: str) -> None:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    print(f"\n[chat] Model: {model}")
    print("[chat] Type 'exit' or press Ctrl-C to quit.\n")
    print("─" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[chat] Goodbye!")
            break

        if user_input.lower() in {"exit", "quit", "q"}:
            print("[chat] Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        response_text = ""
        try:
            stream = ollama.chat(model=model, messages=messages, stream=True)
            for chunk in stream:
                token = chunk["message"]["content"]
                print(token, end="", flush=True)
                response_text += token
        except ollama.ResponseError as exc:
            print(f"\n[error] {exc}")
            print(f"        Try: ollama pull {model}")
            messages.pop()  # remove the user turn we just added
            continue

        print()  # newline after streamed response
        messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a local Ollama model")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model name (default: llama3.2:3b)")
    parser.add_argument("--system", default=SYSTEM_DEFAULT, help="System prompt")
    args = parser.parse_args()

    check_server()
    chat(args.model, args.system)
