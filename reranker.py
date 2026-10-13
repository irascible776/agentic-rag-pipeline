# reranker.py - High-Precision CPU Cross-Encoder Re-Ranking via FlashRank
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Any

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config
from langchain_core.documents import Document

_ranker_instance: Any = None


def get_ranker() -> Any:
    """Lazy singleton initializer for FlashRank cross-encoder model."""
    global _ranker_instance
    if _ranker_instance is None:
        try:
            from flashrank import Ranker
            _ranker_instance = Ranker(model_name=config.RERANKER_MODEL)
            print(f"[Reranker] Initialized FlashRank model '{config.RERANKER_MODEL}' on CPU.")
        except Exception as e:
            print(f"[Reranker] Warning initializing FlashRank ({e}). Reranking will be bypassed.")
            _ranker_instance = False
    return _ranker_instance if _ranker_instance is not False else None


def rerank_documents(
    query: str,
    candidate_docs: List[Document],
    top_n: int = 5
) -> Tuple[List[Document], int]:
    """
    Two-Stage Cross-Encoder Re-Ranking:
    Scores candidate documents against the query simultaneously via full cross-attention.
    Returns: (reranked_docs, latency_ms)
    """
    if not candidate_docs:
        return [], 0

    if not config.ENABLE_RERANKER or len(candidate_docs) <= 1:
        return candidate_docs[:top_n], 0

    ranker: Any = get_ranker()
    if ranker is None:
        # Fallback to candidate order if ranker failed to load
        return candidate_docs[:top_n], 0

    try:
        from flashrank import RerankRequest

        t_start = time.perf_counter()

        # Build passage payloads with index back-references
        passages = [
            {"id": i, "text": doc.page_content}
            for i, doc in enumerate(candidate_docs)
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_req)

        t_end = time.perf_counter()
        rerank_ms = int((t_end - t_start) * 1000)

        # Map scored results back to Document objects
        reranked_docs: List[Document] = []
        for r in results[:top_n]:
            orig_doc = candidate_docs[r["id"]]
            meta = dict(orig_doc.metadata)
            meta["reranker_score"] = float(r["score"])
            reranked_docs.append(Document(
                page_content=orig_doc.page_content,
                metadata=meta
            ))

        return reranked_docs, rerank_ms

    except Exception as e:
        print(f"[Reranker] Error during cross-encoder execution: {e}")
        return candidate_docs[:top_n], 0
