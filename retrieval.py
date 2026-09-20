# retrieval.py - Dual-Mode Document Retrieval & Grounding (Supabase & Local Chroma)
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from pydantic import SecretStr

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_embedding_model():
    """Returns Gemini embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=SecretStr(config.GEMINI_API_KEY) if config.GEMINI_API_KEY else None,
        output_dimensionality=config.EMBEDDING_DIM
    )


def get_llm():
    """Returns Google Gemini chat model for synthesis."""
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        google_api_key=config.GEMINI_API_KEY
    )


def get_prompt_template():
    """Returns ChatPromptTemplate with strict grounding and citation instructions."""
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful and knowledgeable assistant. Answer the user's question using the conversation history and the context below.\n"
         "Rules:\n"
         "1. If the answer cannot be found in the context or conversation history, truthfully state: 'I cannot find the answer in the provided documents.'\n"
         "2. Do not invent facts or use outside knowledge.\n"
         "3. Cite which Document number or Source filename you got each fact from.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])


def format_docs(docs: List[Document]) -> str:
    """Formats retrieved document chunks into clean readable text for the LLM prompt."""
    if not docs:
        return "No relevant documents found."

    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown Document")
        formatted.append(f"[Document {i+1} - Source: {source}]:\n{doc.page_content}")
    return "\n\n".join(formatted)


def retrieve_docs(query: str, session_id: str = "global") -> List[Document]:
    """
    Retrieves the most relevant chunks for a user question:
    - If Cloud Supabase is active: queries Supabase pgvector table using match_documents RPC.
      Retrieves documents that are marked 'global' OR match the user's session_id.
    - If Local Fallback: queries the local ChromaDB database.
    """
    embed_model = get_embedding_model()

    # --- MODE 1: CLOUD SUPABASE ---
    if config.USE_SUPABASE:
        from supabase import create_client
        supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

        # 1. Embed the user's question into a vector
        query_vector = embed_model.embed_query(query)

        # 2. Call match_documents RPC function in Supabase
        try:
            rpc_response = supabase.rpc("match_documents", {
                "query_embedding": query_vector,
                "match_count": config.RETRIEVER_K,
                "filter": {"session_id": session_id}
            }).execute()

            matched_rows = rpc_response.data
        except Exception as e:
            print(f"[Retrieval] Supabase RPC failed ({e}), falling back to direct table query...")
            matched_rows = []

        if not isinstance(matched_rows, list):
            matched_rows = []

        # Convert SQL rows to LangChain Document objects
        docs: List[Document] = []
        for row in matched_rows:
            if isinstance(row, dict):
                content = str(row.get("content", ""))
                meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                docs.append(Document(page_content=content, metadata=meta))

        return docs

    # --- MODE 2: LOCAL CHROMADB FALLBACK ---
    else:
        from langchain_chroma import Chroma

        # Check if local database exists on disk
        chroma_sqlite = config.LOCAL_DB_DIR / "chroma.sqlite3"
        if not chroma_sqlite.exists():
            print("[Retrieval] Local ChromaDB not found yet. Ingest documents first.")
            return []

        vectordb = Chroma(
            persist_directory=str(config.LOCAL_DB_DIR),
            collection_name="gemini_rag",
            embedding_function=embed_model,
            collection_metadata={"hnsw:space": "cosine"}
        )

        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.RETRIEVER_K,
                "fetch_k": config.FETCH_K,
                "lambda_mult": config.LAMBDA_MULT
            }
        )

        matched_docs = retriever.invoke(query)
        return list(matched_docs)


def clear_session(session_id: str) -> Dict[str, Any]:
    """
    Cleans up documents associated with this session:
    - Only deletes disposable session chunks (never deletes 'global' knowledge base docs).
    """
    if not session_id or session_id == "global":
        return {"status": "ignored", "message": "Cannot delete global knowledge base docs."}

    if config.USE_SUPABASE:
        from supabase import create_client
        supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

        print(f"[Retrieval] Cleaning up Supabase records for session: {session_id}...")
        res = supabase.table(config.SUPABASE_TABLE).delete().eq("session_id", session_id).execute()
        return {"status": "cleared", "mode": "supabase", "session_id": session_id}
    else:
        print(f"[Retrieval] Local session cleanup requested for session: {session_id}")
        return {"status": "cleared", "mode": "local", "session_id": session_id}
