"""
04_rag_basic.py
───────────────
Retrieval-Augmented Generation (RAG) with LangChain + ChromaDB.

Demonstrates:
  - Splitting documents into chunks
  - Embedding chunks with sentence-transformers (local, no API key needed)
  - Storing embeddings in an in-memory ChromaDB vector store
  - Retrieval-augmented Q&A using an Ollama LLM as the generator

Architecture:
  Documents → Splitter → Embeddings → ChromaDB
                                          ↓
  Question → Embed → Retrieve top-k chunks → LLM → Answer

Prerequisites:
    ollama pull llama3.2:3b      # or another model of your choice

Usage:
    python 04_rag_basic.py
    python 04_rag_basic.py --llm mistral:7b --top-k 4
"""

import argparse
import textwrap

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ── Sample knowledge base ─────────────────────────────────────
DOCUMENTS = [
    Document(
        page_content=(
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "information retrieval with generative language models. Instead of relying "
            "solely on the model's parametric knowledge, RAG retrieves relevant passages "
            "from an external corpus at query time and provides them as context to the "
            "generator. This reduces hallucinations and keeps responses factual."
        ),
        metadata={"source": "rag_intro.txt"},
    ),
    Document(
        page_content=(
            "ChromaDB is an open-source vector database optimised for AI applications. "
            "It stores embeddings alongside metadata and supports fast approximate nearest "
            "neighbour search. ChromaDB can run fully in-process (no server required) or "
            "as a standalone server, making it easy to prototype RAG pipelines locally."
        ),
        metadata={"source": "chroma_overview.txt"},
    ),
    Document(
        page_content=(
            "LangChain is a framework for building applications powered by language models. "
            "It provides composable abstractions for chains, agents, tools, memory, and "
            "document loaders. Its integration library (langchain-community) contains "
            "connectors for dozens of vector stores, LLMs, and data sources."
        ),
        metadata={"source": "langchain_overview.txt"},
    ),
    Document(
        page_content=(
            "Ollama is a tool for running large language models locally. It supports "
            "models in the GGUF format such as Llama 3, Mistral, Gemma, and Phi. "
            "Ollama exposes an OpenAI-compatible REST API on port 11434, so any client "
            "that speaks the OpenAI chat completions protocol can use it."
        ),
        metadata={"source": "ollama_overview.txt"},
    ),
    Document(
        page_content=(
            "Sentence-transformers is a Python library for computing dense vector "
            "representations (embeddings) of sentences and paragraphs. Models like "
            "'all-MiniLM-L6-v2' and 'BAAI/bge-small-en-v1.5' produce high-quality "
            "embeddings suitable for semantic search, clustering, and RAG retrieval."
        ),
        metadata={"source": "sentence_transformers.txt"},
    ),
    Document(
        page_content=(
            "FAISS (Facebook AI Similarity Search) is a library for efficient similarity "
            "search over dense vectors. It supports both exact and approximate nearest "
            "neighbour algorithms and can scale to billions of vectors with GPU acceleration."
        ),
        metadata={"source": "faiss_overview.txt"},
    ),
]

QUESTIONS = [
    "What is Retrieval-Augmented Generation?",
    "How does ChromaDB work?",
    "Which models does Ollama support?",
    "What embedding models work well for semantic search?",
]

PROMPT_TEMPLATE = """Use the following retrieved context to answer the question.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""


def build_vectorstore(
    docs: list[Document],
    embedding_model: str,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"[rag] Split {len(docs)} documents into {len(chunks)} chunks")

    print(f"[rag] Embedding with '{embedding_model}' ...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    vectorstore = Chroma.from_documents(chunks, embeddings)
    print(f"[rag] Vector store ready ({vectorstore._collection.count()} vectors)")
    return vectorstore


def main(llm_model: str, embedding_model: str, top_k: int) -> None:
    vectorstore = build_vectorstore(DOCUMENTS, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    print(f"[rag] Connecting to Ollama LLM '{llm_model}' ...")
    llm = Ollama(model=llm_model, temperature=0.1)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    print(f"\n{'─' * 60}")
    for question in QUESTIONS:
        print(f"\n[Q] {question}")
        result = qa_chain.invoke({"query": question})
        answer = result["result"].strip()
        sources = {doc.metadata.get("source", "?") for doc in result["source_documents"]}
        print(f"[A] {textwrap.fill(answer, width=68, subsequent_indent='    ')}")
        print(f"[sources] {', '.join(sorted(sources))}")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG demo with LangChain + ChromaDB + Ollama")
    parser.add_argument("--llm", default="llama3.2:3b", help="Ollama model (default: llama3.2:3b)")
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model for embeddings",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")
    args = parser.parse_args()
    main(args.llm, args.embedding_model, args.top_k)
