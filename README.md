# agentic-rag-pipeline
Hands-on implementation of an end-to-end RAG system: document ingestion, local vector embeddings, MMR retrieval, and hallucination-guarded LLM generation.

# Local-First RAG Pipeline

A robust, end-to-end Retrieval-Augmented Generation (RAG) pipeline built with LangChain, featuring **zero-rate-limit local embeddings** and **grounded LLM synthesis** with citations.

---

## 🚀 Architecture

* **Document Ingestion (`ingestion.py`)**:
  * Loads documents from `docs/` using `DirectoryLoader`.
  * Splits text into semantic chunks using `RecursiveCharacterTextSplitter`.
  * Generates local 384-dimensional vector embeddings using **`BAAI/bge-small-en-v1.5`** (runs 100% offline on CPU with zero rate limits).
  * Stores embeddings in a persistent local **ChromaDB** vector store.

* **Retrieval & Answer Generation (`retrieval.py`)**:
  * Performs **Maximal Marginal Relevance (MMR)** similarity search to balance relevance and diversity.
  * Formats retrieved chunks with source attribution metadata.
  * Prompts **Gemini 3.6 Flash** with strict anti-hallucination guardrails and document citation rules.

* **Embedding Benchmarks (`test.py`)**:
  * Side-by-side benchmark comparing `BAAI/bge-small-en-v1.5` (512 tokens) and `nomic-ai/nomic-embed-text-v1.5` (8,192 tokens) on CPU inference speed and cosine similarity spread.

---


