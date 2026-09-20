# Agentic RAG Pipeline

A modular, production-ready Retrieval-Augmented Generation (RAG) system built with LangChain, FastAPI, and Google Gemini. It supports multi-format document ingestion (.pdf, .docx, .txt, .md), dual-mode vector storage (Supabase pgvector and local ChromaDB), conversation history awareness, and a clean web interface.

---

## Overview

Most beginner RAG tutorials rely on a static Jupyter Notebook that embeds a few text files into an ephemeral vector store and queries them once. In real applications, systems need to ingest diverse document types on the fly, isolate user sessions, handle multi-turn conversations, adapt between local testing and cloud deployment, and maintain tight control over memory usage.

This project implements an end-to-end RAG architecture designed to address those practical engineering challenges:

- Multi-Format Document Ingestion: Extracts and chunks text from PDF, DOCX, TXT, and Markdown files in real time.
- Zero-RAM Cloud Embedding Strategy: Uses Google's gemini-embedding-001 with configured 768 output dimensions. Vector math runs via cloud API, preventing the out-of-memory crashes common when loading local PyTorch embedding models on memory-constrained servers.
- Dual-Mode Vector Storage: Automatically checks environment configuration. If Supabase credentials are present, vectors and metadata persist to PostgreSQL with pgvector. If omitted, the system falls back to an embedded ChromaDB store without requiring any external database setup.
- Session-Isolated Conversational Memory: Maintains chat history per session ID. Follow-up queries understand context from prior exchanges while grounding answers strictly in the retrieved documents.
- Hallucination Guardrails: Prompts enforce strict source attribution. The model explicitly cites source documents and states when information is missing from the indexed context rather than hallucinating.
- Lightweight Web Interface: A clean, responsive single-page application and FastAPI backend for uploading files, monitoring indexed chunks, and chatting in real time.

---

## Architecture and Workflow

The pipeline consists of three core stages:

1. Ingestion (ingestion.py):
   Incoming files (either from the docs/ directory or uploaded through the web UI) are inspected by file extension. The parser extracts plain text using pypdf for PDF files, python-docx for Word documents, and text decoders for Markdown and plain text. Text is split into 1500-character chunks with a 150-character overlap using recursive character splitting. Chunks are embedded and stored in the active vector store tagged with document metadata and session IDs.

2. Retrieval and Grounding (retrieval.py):
   When a query arrives, the system queries the vector database using vector similarity search to find the top matching chunks (k = 5). Chunks are formatted with source labels and passed to the generation layer.

3. Context-Aware Synthesis (chat.py):
   The retrieved document snippets, previous conversation turns, and the user's current question are formatted into a structured prompt. Gemini (gemini-3.5-flash) generates a response citing the sources used. If the document chunks do not contain the answer, the model acknowledges that the information is unavailable.

---

## Project Structure

```
agentic-rag-pipeline/
├── config.py             # Centralized settings and environment auto-detection
├── requirements.txt      # Python dependencies
├── .env.example          # Template for environment variables
├── .gitignore            # Git exclusion rules
├── ingestion.py          # Multi-format parsing, chunking, and vector ingestion
├── retrieval.py          # Vector search, MMR, prompt templates, and DB management
├── chat.py               # Conversational synthesis with chat history support
├── __init__.py           # Standardized 3-function interface (ingest, ask, clear)
├── test.py               # Benchmark script comparing local embedding models
├── docs/                 # Default sample documents
└── gui/
    ├── server.py         # FastAPI REST API serving the web app and pipeline
    └── static/
        ├── index.html    # Studio user interface layout
        ├── app.js        # Client logic, file staging, and chat streaming
        └── style.css     # Clean typography and responsive design system
```

---

## Quickstart Guide

### 1. Prerequisites

- Python 3.10, 3.11, or 3.12
- A Google Gemini API key (available from Google AI Studio)

### 2. Clone and Setup Environment

Clone the repository and set up a virtual environment:

```bash
git clone git@github.com:irascible776/agentic-rag-pipeline.git
cd agentic-rag-pipeline

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a .env file by copying the example template:

```bash
cp .env.example .env
```

Open .env and add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

To enable Supabase pgvector storage, add your database credentials. If left blank, the application automatically uses local ChromaDB:

```env
SUPABASE_URL=
SUPABASE_KEY=
```

### 4. Run the Web Studio

Start the FastAPI application:

```bash
python gui/server.py
```

Alternatively, run with Uvicorn directly:

```bash
uvicorn gui.server:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at http://127.0.0.1:8000 to interact with the web interface.

---

## Web Studio Features

- Drag and Drop Uploads: Add multiple .pdf, .docx, .txt, or .md files at once.
- Chunk Counter: View the number of indexed text chunks currently active in the vector database.
- Document Badges: See exactly which uploaded documents have been parsed and indexed.
- Reset Knowledge Base: Clear active session documents and conversation history with a single click.
- Conversational Context: Ask questions, receive cited answers, and ask follow-up questions referencing prior turns.

---

## API Reference

The FastAPI server exposes the following REST endpoints:

### GET /api/config
Returns the active configuration status, including whether Supabase or local Chroma is in use, the LLM model name, and the embedding model.

### POST /api/upload-chunk
Accepts multipart/form-data uploads containing one or more files and a session_id. Parses documents, chunks text, creates embeddings, and updates the vector store.

### POST /api/chat
Accepts a JSON payload:
```json
{
  "question": "What are the key findings in section 2?",
  "history": [
    {"role": "user", "content": "What is the document about?"},
    {"role": "assistant", "content": "The document discusses..."}
  ],
  "session_id": "global"
}
```
Returns:
```json
{
  "answer": "Section 2 outlines the primary methodology...",
  "sources": ["report.pdf"]
}
```

### POST /api/session/clear
Clears stored documents and conversation state for the specified session_id.

---

## Programmatic Usage

You can also import and use the pipeline directly inside your own Python code:

```python
from ingestion import ingest_documents, extract_text_from_bytes
from chat import ask_question
from retrieval import clear_session

# Ingest documents from a directory
result = ingest_documents("docs/", session_id="user_123")
print(f"Indexed {result['chunks_count']} chunks.")

# Ask questions with chat history
history = []
response = ask_question("Summarize the main topic", history=history, session_id="user_123")
print("Answer:", response["answer"])
print("Sources:", response["sources"])

# Maintain history for follow-up
history.append({"role": "user", "content": "Summarize the main topic"})
history.append({"role": "assistant", "content": response["answer"]})

follow_up = ask_question("Tell me more about the first point", history=history, session_id="user_123")
print("Follow-up Answer:", follow_up["answer"])

# Clear session when finished
clear_session("user_123")
```

---

## Configuration Options

All pipeline settings can be adjusted in config.py or overridden using environment variables:

| Setting | Default Value | Description |
| :--- | :--- | :--- |
| GEMINI_API_KEY | None | API key for Gemini models |
| SUPABASE_URL | None | Supabase project URL for pgvector |
| SUPABASE_KEY | None | Supabase service key |
| LLM_MODEL | gemini-3.5-flash | Gemini model used for synthesis |
| LLM_TEMPERATURE | 0.1 | Sampling temperature for factual accuracy |
| EMBEDDING_MODEL | gemini-embedding-001 | Cloud embedding model |
| EMBEDDING_DIM | 768 | Embedding dimensionality |
| CHUNK_SIZE | 1500 | Target character length per chunk |
| CHUNK_OVERLAP | 150 | Overlap between adjacent chunks |
| RETRIEVER_K | 5 | Number of chunks retrieved per query |

---

## License

MIT
