"""
01_basic_llm_inference.py
─────────────────────────
Text generation with HuggingFace Transformers.

Demonstrates:
  - Loading a small causal LM (GPT-2 by default, no auth required)
  - Greedy and sampling-based decoding
  - Device detection: uses CUDA when available, else CPU

Usage:
    python 01_basic_llm_inference.py
    python 01_basic_llm_inference.py --model "facebook/opt-350m"
"""

import argparse
import textwrap

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[device] CUDA — {name} ({mem:.1f} GB VRAM)")
    else:
        device = "cpu"
        print("[device] CPU (no CUDA GPU detected)")
    return device


def main(model_name: str, prompt: str) -> None:
    device = get_device()

    print(f"\n[model] Loading '{model_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
    )

    print(f"\n[prompt] {prompt}\n")
    print("─" * 60)

    # Greedy decoding
    result_greedy = generator(
        prompt,
        max_new_tokens=80,
        do_sample=False,
    )
    print("[greedy]\n" + textwrap.fill(result_greedy[0]["generated_text"], width=72))
    print()

    # Sampling with temperature
    result_sample = generator(
        prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
    )
    print("[sampling (T=0.8)]\n" + textwrap.fill(result_sample[0]["generated_text"], width=72))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic LLM inference with HuggingFace Transformers")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model ID (default: gpt2)")
    parser.add_argument(
        "--prompt",
        default="Artificial intelligence is transforming the world because",
        help="Input prompt",
    )
    args = parser.parse_args()
    main(args.model, args.prompt)
