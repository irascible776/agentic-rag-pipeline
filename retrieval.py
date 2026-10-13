# retrieval.py - Dual-Mode Document Retrieval & Grounding (Supabase & Local Chroma)
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
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


def get_llm(model: str | None = None, temperature: float | None = None):
    """Returns Google Gemini chat model for synthesis with optional overrides."""
    return ChatGoogleGenerativeAI(
        model=model or config.LLM_MODEL,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
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


def resolve_hierarchical_context(docs: List[Document]) -> List[Document]:
    """
    Parent-Child Context Resolver:
    Deduplicates child chunks that belong to the same parent section, and expands
    each chunk to its full rich parent context for high-accuracy LLM synthesis.
    """
    seen_parents = set()
    resolved: List[Document] = []

    for doc in docs:
        parent_id = doc.metadata.get("parent_id")
        parent_content = doc.metadata.get("parent_content")

        if parent_id and parent_content:
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            # Create expanded parent document while preserving metadata
            resolved.append(Document(page_content=parent_content, metadata=doc.metadata))
        else:
            resolved.append(doc)

    return resolved


def extract_media_citations(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Extracts unique media citations (images, diagrams) referenced in retrieved documents.
    """
    media_citations = []
    seen_media = set()

    for doc in docs:
        media_path = doc.metadata.get("media_path")
        media_url = doc.metadata.get("media_url")
        doc_type = doc.metadata.get("type")

        if media_path or media_url or doc_type == "image":
            url = media_url or (f"/media/{media_path}" if media_path else "")
            if not url or url in seen_media:
                continue
            seen_media.add(url)

            caption = doc.metadata.get("media_caption")
            if not caption:
                caption = doc.page_content.split("\n")[0][:120]

            media_citations.append({
                "url": url,
                "type": "image",
                "caption": caption,
                "source": doc.metadata.get("source", "Document"),
                "page": doc.metadata.get("page", 1),
                "media_path": media_path
            })

    return media_citations


def format_docs(docs: List[Document]) -> str:
    """Formats retrieved document chunks into clean readable text for the LLM prompt."""
    if not docs:
        return "No relevant documents found."

    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page")
        page_str = f" (Page {page})" if page else ""
        doc_type = doc.metadata.get("type", "text")
        type_str = f" [{doc_type.upper()}]" if doc_type != "text" else ""
        formatted.append(f"[Document {i+1} - Source: {source}{page_str}{type_str}]:\n{doc.page_content}")
    return "\n\n".join(formatted)



def _run_dense_vector_search(
    query: str,
    session_id: str = "global",
    fetch_k: int = 25
) -> List[Document]:
    """Retrieves candidate passages from dense vector database (Supabase or Chroma)."""
    embed_model = get_embedding_model()

    # --- MODE 1: CLOUD SUPABASE ---
    if config.USE_SUPABASE:
        try:
            from supabase import create_client
            supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

            query_vector = embed_model.embed_query(query)
            rpc_response = supabase.rpc("match_documents", {
                "query_embedding": query_vector,
                "match_count": fetch_k,
                "filter": {"session_id": session_id}
            }).execute()

            matched_rows = rpc_response.data
            if isinstance(matched_rows, list) and len(matched_rows) > 0:
                docs: List[Document] = []
                for row in matched_rows:
                    if isinstance(row, dict):
                        content = str(row.get("content", ""))
                        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                        docs.append(Document(page_content=content, metadata=meta))
                return docs
        except Exception as e:
            print(f"[Retrieval] Supabase RPC failed/inaccessible ({e}), falling back to local ChromaDB...")

    # --- MODE 2: LOCAL CHROMADB FALLBACK ---
    from langchain_chroma import Chroma

    chroma_sqlite = config.LOCAL_DB_DIR / "chroma.sqlite3"
    if not chroma_sqlite.exists():
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
            "k": fetch_k,
            "fetch_k": max(config.FETCH_K, fetch_k * 2),
            "lambda_mult": config.LAMBDA_MULT
        }
    )

    matched_docs = retriever.invoke(query)
    return list(matched_docs)


def retrieve_docs_with_telemetry(
    query: str,
    session_id: str = "global",
    k: int | None = None,
    enable_hybrid: bool | None = None,
    enable_reranker: bool | None = None
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Level 3 Two-Stage Retrieval Pipeline:
    1. Dense Vector Search (semantic embeddings)
    2. Sparse BM25 Search (exact keywords & codes)
    3. Reciprocal Rank Fusion (RRF score merging)
    4. Two-Stage Cross-Encoder Re-Ranking (FlashRank CPU ONNX)
    Returns: (final_documents, latency_telemetry_dict)
    """
    import time
    use_hybrid = enable_hybrid if enable_hybrid is not None else config.ENABLE_HYBRID_SEARCH
    use_reranker = enable_reranker if enable_reranker is not None else config.ENABLE_RERANKER

    retrieval_k = k if k and k > 0 else config.RETRIEVER_K
    fetch_k = config.HYBRID_FETCH_K if (use_hybrid or use_reranker) else retrieval_k

    # 1. Dense Vector Search
    t_v0 = time.perf_counter()
    dense_candidates = _run_dense_vector_search(query, session_id=session_id, fetch_k=fetch_k)
    t_v1 = time.perf_counter()
    dense_ms = int((t_v1 - t_v0) * 1000)

    # 2. Sparse Lexical Search (BM25) & Reciprocal Rank Fusion
    bm25_ms = 0
    if use_hybrid:
        t_b0 = time.perf_counter()
        from hybrid_search import search_bm25, reciprocal_rank_fusion
        sparse_candidates = search_bm25(query, session_id=session_id, k=fetch_k)
        t_b1 = time.perf_counter()
        bm25_ms = int((t_b1 - t_b0) * 1000)

        fused_candidates = reciprocal_rank_fusion(
            dense_docs=dense_candidates,
            sparse_docs=sparse_candidates,
            k_const=config.RRF_SMOOTHING,
            top_n=fetch_k
        )
    else:
        fused_candidates = dense_candidates

    # 3. Two-Stage Cross-Encoder Re-Ranking (FlashRank)
    rerank_ms = 0
    if use_reranker and fused_candidates:
        from reranker import rerank_documents
        final_docs, rerank_ms = rerank_documents(query, fused_candidates, top_n=retrieval_k)
    else:
        final_docs = fused_candidates[:retrieval_k]

    telemetry = {
        "dense_search_ms": dense_ms,
        "bm25_search_ms": bm25_ms,
        "rerank_ms": rerank_ms,
        "candidates_pooled": len(fused_candidates),
        "final_chunks": len(final_docs),
        "hybrid_enabled": use_hybrid,
        "reranker_enabled": use_reranker
    }

    return final_docs, telemetry


def retrieve_docs(
    query: str,
    session_id: str = "global",
    k: int | None = None,
    enable_hybrid: bool | None = None,
    enable_reranker: bool | None = None
) -> List[Document]:
    """Standard backwards-compatible retrieval entrypoint."""
    docs, _ = retrieve_docs_with_telemetry(
        query,
        session_id=session_id,
        k=k,
        enable_hybrid=enable_hybrid,
        enable_reranker=enable_reranker
    )
    return docs


def clear_session(session_id: str) -> Dict[str, Any]:
    """
    Cleans up documents associated with this session:
    - Deletes session vectors from Supabase or Chroma.
    - Cleans up session entries from BM25 index.
    - Leaves permanent/global knowledge base docs intact.
    """
    if not session_id or session_id == "global":
        return {"status": "ignored", "message": "Cannot delete global knowledge base docs."}

    # Clean up BM25 index
    try:
        from hybrid_search import get_bm25_manager
        get_bm25_manager().clear_session(session_id)
    except Exception as e:
        print(f"[Retrieval] Warning clearing BM25 session {session_id}: {e}")

    if config.USE_SUPABASE:
        try:
            from supabase import create_client
            supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

            print(f"[Retrieval] Cleaning up Supabase records for session: {session_id}...")
            supabase.table(config.SUPABASE_TABLE).delete().eq("session_id", session_id).execute()
            return {"status": "cleared", "mode": "supabase", "session_id": session_id}
        except Exception as e:
            print(f"[Retrieval] Supabase session clear warning ({e}), falling back to local...")

    print(f"[Retrieval] Local session cleanup completed for session: {session_id}")
    return {"status": "cleared", "mode": "local", "session_id": session_id}


