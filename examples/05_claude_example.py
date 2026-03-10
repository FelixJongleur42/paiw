"""
05_claude_example.py
────────────────────
Conversational exchange and streaming with Anthropic's Claude API.

Demonstrates:
  - Single-turn completion
  - Multi-turn conversation
  - Streaming response tokens
  - Structured output (JSON mode via system prompt)

Prerequisites:
    Set ANTHROPIC_API_KEY in your .env file or as an environment variable.

Usage:
    python 05_claude_example.py
    python 05_claude_example.py --model claude-3-haiku-20240307
    python 05_claude_example.py --interactive
"""

import argparse
import json
import os
import sys
import textwrap

import anthropic


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[error] ANTHROPIC_API_KEY is not set.")
        print("        Add it to the .env file at the repository root, e.g.:")
        print("          ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def demo_single_turn(client: anthropic.Anthropic, model: str) -> None:
    print("\n── Single-turn completion ───────────────────────────────────")
    message = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": "In two sentences, explain what Retrieval-Augmented Generation is.",
            }
        ],
    )
    print(textwrap.fill(message.content[0].text, width=72))
    print(f"[usage] input={message.usage.input_tokens} tokens, output={message.usage.output_tokens} tokens")


def demo_streaming(client: anthropic.Anthropic, model: str) -> None:
    print("\n── Streaming response ───────────────────────────────────────")
    prompt = "List five practical use-cases for large language models in enterprise software."
    print(f"[prompt] {prompt}\n")
    with client.messages.stream(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()


def demo_multi_turn(client: anthropic.Anthropic, model: str) -> None:
    print("\n── Multi-turn conversation ──────────────────────────────────")
    turns = [
        "What is a vector database?",
        "Which open-source vector databases are most popular?",
        "How does ChromaDB compare to the others you mentioned?",
    ]
    messages: list[dict] = []
    for user_text in turns:
        print(f"\nUser:      {user_text}")
        messages.append({"role": "user", "content": user_text})
        response = client.messages.create(
            model=model,
            max_tokens=300,
            system="You are a concise technical assistant. Keep answers under 4 sentences.",
            messages=messages,
        )
        assistant_text = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_text})
        print(f"Assistant: {textwrap.fill(assistant_text, width=68, subsequent_indent='           ')}")


def demo_structured_output(client: anthropic.Anthropic, model: str) -> None:
    print("\n── Structured output (JSON) ─────────────────────────────────")
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=(
            "You are a JSON API. Always respond with valid JSON only, no prose. "
            "Return an object with keys: 'name', 'description', 'use_cases' (array of strings)."
        ),
        messages=[
            {"role": "user", "content": "Describe the FAISS library."}
        ],
    )
    raw = response.content[0].text.strip()
    try:
        data = json.loads(raw)
        print(json.dumps(data, indent=2))
    except json.JSONDecodeError:
        print(f"[warn] Could not parse JSON, raw response:\n{raw}")


def interactive_chat(client: anthropic.Anthropic, model: str) -> None:
    print(f"\n[chat] Model: {model}")
    print("[chat] Type 'exit' to quit.\n")
    messages: list[dict] = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[chat] Goodbye!")
            break
        if user_input.lower() in {"exit", "quit", "q"}:
            print("[chat] Goodbye!")
            break
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        print("Claude: ", end="", flush=True)
        with client.messages.stream(
            model=model,
            max_tokens=1024,
            messages=messages,
        ) as stream:
            response_text = ""
            for token in stream.text_stream:
                print(token, end="", flush=True)
                response_text += token
        print()
        messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude API demo")
    parser.add_argument(
        "--model",
        default="claude-3-haiku-20240307",
        help="Anthropic model ID (default: claude-3-haiku-20240307)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive chat session instead of running the demos",
    )
    args = parser.parse_args()

    client = get_client()

    if args.interactive:
        interactive_chat(client, args.model)
    else:
        demo_single_turn(client, args.model)
        demo_streaming(client, args.model)
        demo_multi_turn(client, args.model)
        demo_structured_output(client, args.model)
