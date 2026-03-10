"""
03_embeddings.py
────────────────
Sentence embeddings and semantic similarity search with sentence-transformers.

Demonstrates:
  - Encoding sentences into dense vectors
  - Cosine-similarity ranking
  - Simple nearest-neighbour search with FAISS
  - Device detection (CUDA / CPU)

Usage:
    python 03_embeddings.py
    python 03_embeddings.py --model "BAAI/bge-small-en-v1.5"
"""

import argparse

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


CORPUS = [
    "The Eiffel Tower is located in Paris, France.",
    "Python is a popular programming language for machine learning.",
    "Large language models are trained on vast amounts of text data.",
    "Retrieval-Augmented Generation combines search with text generation.",
    "Neural networks consist of layers of interconnected nodes.",
    "The Amazon rainforest is the world's largest tropical rainforest.",
    "Transformers use self-attention mechanisms to process sequences.",
    "Docker containers provide a consistent runtime environment.",
    "Vector databases store and query high-dimensional embeddings efficiently.",
    "Fine-tuning adapts a pre-trained model to a specific task or domain.",
]

QUERIES = [
    "How do attention mechanisms work?",
    "What is RAG in AI?",
    "Where is the Eiffel Tower?",
    "How do I run code in containers?",
]


def cosine_search(model: SentenceTransformer, corpus_emb: np.ndarray, top_k: int = 3) -> None:
    """Brute-force cosine similarity search using sentence-transformers utilities."""
    print("\n── Cosine similarity search ────────────────────────────────")
    for query in QUERIES:
        query_emb = model.encode(query, convert_to_tensor=True)
        corpus_tensor = torch.tensor(corpus_emb)
        hits = util.semantic_search(query_emb, corpus_tensor, top_k=top_k)[0]
        print(f"\nQuery: {query!r}")
        for hit in hits:
            print(f"  [{hit['score']:.3f}] {CORPUS[hit['corpus_id']]}")


def faiss_search(model: SentenceTransformer, corpus_emb: np.ndarray, top_k: int = 3) -> None:
    """FAISS inner-product index (equivalent to cosine similarity on L2-normalised vectors)."""
    print("\n── FAISS nearest-neighbour search ──────────────────────────")
    dim = corpus_emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on L2-normalised vectors == cosine similarity
    index.add(corpus_emb.astype(np.float32))
    print(f"[faiss] Index contains {index.ntotal} vectors of dimension {dim}")

    query_emb = model.encode(QUERIES, normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(query_emb, k=top_k)

    for i, query in enumerate(QUERIES):
        print(f"\nQuery: {query!r}")
        for dist, idx in zip(distances[i], indices[i]):
            print(f"  [{dist:.3f}] {CORPUS[idx]}")


def main(model_name: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device.upper()}")
    print(f"[model]  Loading '{model_name}' ...")

    model = SentenceTransformer(model_name, device=device)

    print(f"[encode] Encoding {len(CORPUS)} corpus sentences ...")
    corpus_emb = model.encode(CORPUS, normalize_embeddings=True, show_progress_bar=True)
    print(f"[encode] Embedding shape: {corpus_emb.shape}")

    cosine_search(model, corpus_emb)
    faiss_search(model, corpus_emb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentence embeddings and similarity search")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()
    main(args.model)
