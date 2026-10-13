# __init__.py - Clean 3-Function Pipeline Interface
"""
Standard RAG Pipeline Contract:
Any pipeline in this project exposes these 3 core functions:
1. ingest(files_or_folder, session_id, is_permanent) -> dict
2. ask(question, history, session_id) -> dict
3. clear(session_id) -> dict
"""

from ingestion import ingest_documents as ingest, extract_text_from_bytes
from chat import ask_question as ask
from retrieval import clear_session as clear, retrieve_docs, retrieve_docs_with_telemetry
import media_store
import vision
import hybrid_search
import reranker

__all__ = [
    "ingest",
    "ask",
    "clear",
    "retrieve_docs",
    "retrieve_docs_with_telemetry",
    "extract_text_from_bytes",
    "media_store",
    "vision",
    "hybrid_search",
    "reranker",
]

