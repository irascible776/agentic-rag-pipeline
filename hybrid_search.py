# hybrid_search.py - Sparse BM25 Lexical Index & Reciprocal Rank Fusion (RRF)
import os
import sys
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """Fast regex alphanumeric tokenizer."""
    return re.findall(r"\w+", text.lower())


class BM25IndexManager:
    """
    Manages in-memory and disk-persisted BM25 inverted index for exact keyword search.
    Preserves document metadata and session tags.
    """
    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = index_path or config.BM25_INDEX_PATH
        self.documents: List[Dict[str, Any]] = []  # list of {"page_content": str, "metadata": dict}
        self.bm25: Optional[BM25Okapi] = None
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Loads cached index from disk if present."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data.get("documents", [])
                if self.documents:
                    tokenized = [tokenize(d["page_content"]) for d in self.documents]
                    self.bm25 = BM25Okapi(tokenized)
                print(f"[HybridSearch] Loaded BM25 index with {len(self.documents)} chunks.")
            except Exception as e:
                print(f"[HybridSearch] Warning loading BM25 index ({e}). Starting fresh.")
                self.documents = []
                self.bm25 = None

    def _save_to_disk(self) -> None:
        """Persists index documents to disk."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump({"documents": self.documents}, f)
        except Exception as e:
            print(f"[HybridSearch] Warning saving BM25 index to disk: {e}")

    def add_documents(self, chunks: List[Document], session_id: str = "global") -> None:
        """Adds new Document chunks to the BM25 index and updates the model."""
        if not chunks:
            return

        existing_keys = {
            (d["page_content"], d["metadata"].get("session_id", "global"))
            for d in self.documents
        }

        added_count = 0
        for chunk in chunks:
            meta = dict(chunk.metadata)
            meta["session_id"] = meta.get("session_id", session_id)
            key = (chunk.page_content, meta["session_id"])
            if key not in existing_keys:
                self.documents.append({
                    "page_content": chunk.page_content,
                    "metadata": meta
                })
                existing_keys.add(key)
                added_count += 1

        if added_count > 0:
            tokenized = [tokenize(d["page_content"]) for d in self.documents]
            self.bm25 = BM25Okapi(tokenized)
            self._save_to_disk()
            print(f"[HybridSearch] Indexed {added_count} new chunks into BM25 (Total: {len(self.documents)}).")

    def search(self, query: str, session_id: str = "global", k: int = 25) -> List[Document]:
        """
        Executes sparse BM25 query against indexed chunks.
        Filters to documents matching session_id or 'global'.
        """
        query_tokens = tokenize(query)
        if not query_tokens or not self.bm25 or not self.documents:
            return []

        try:
            scores = self.bm25.get_scores(query_tokens)

            # Filter and pair by matching session_id
            candidates = []
            for idx, score in enumerate(scores):
                if score <= 0.0:
                    continue
                doc_dict = self.documents[idx]
                doc_session = doc_dict["metadata"].get("session_id", "global")
                if doc_session in ("global", session_id):
                    candidates.append((score, doc_dict))

            # Sort descending by score
            candidates.sort(key=lambda x: x[0], reverse=True)

            results: List[Document] = []
            for score, doc_dict in candidates[:k]:
                meta = dict(doc_dict["metadata"])
                meta["bm25_score"] = float(score)
                results.append(Document(
                    page_content=doc_dict["page_content"],
                    metadata=meta
                ))

            return results
        except Exception as e:
            print(f"[HybridSearch] BM25 search warning: {e}")
            return []

    def clear_session(self, session_id: str) -> None:
        """Removes disposable session chunks from BM25 index."""
        if not session_id or session_id == "global":
            return
        initial_len = len(self.documents)
        self.documents = [
            d for d in self.documents
            if d["metadata"].get("session_id") != session_id
        ]
        if len(self.documents) != initial_len:
            if self.documents:
                tokenized = [tokenize(d["page_content"]) for d in self.documents]
                self.bm25 = BM25Okapi(tokenized)
            else:
                self.bm25 = None
            self._save_to_disk()


# Global Singleton Manager
_bm25_manager: Optional[BM25IndexManager] = None


def get_bm25_manager() -> BM25IndexManager:
    global _bm25_manager
    if _bm25_manager is None:
        _bm25_manager = BM25IndexManager()
    return _bm25_manager


def update_bm25_index(chunks: List[Document], session_id: str = "global") -> None:
    """Helper to update BM25 index during ingestion."""
    manager = get_bm25_manager()
    manager.add_documents(chunks, session_id=session_id)


def search_bm25(query: str, session_id: str = "global", k: int = 25) -> List[Document]:
    """Helper to query BM25 index."""
    manager = get_bm25_manager()
    return manager.search(query, session_id=session_id, k=k)


def reciprocal_rank_fusion(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    k_const: int = 60,
    top_n: int = 25
) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) algorithm:
    Merges dense vector ranking and sparse BM25 ranking into a unified candidate pool.
    RRF(d) = sum( 1.0 / (k_const + rank) ) for each ranking where d appears.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    def get_key(doc: Document) -> str:
        # Key by parent_id + content hash or source + page + first 100 chars
        src = doc.metadata.get("source", "doc")
        pid = doc.metadata.get("parent_id", "")
        return f"{src}_{pid}_{doc.page_content[:120]}"

    # 1. Score dense vector documents
    for rank, doc in enumerate(dense_docs):
        key = get_key(doc)
        scores[key] = scores.get(key, 0.0) + (1.0 / (k_const + rank + 1))
        if key not in doc_map:
            doc_map[key] = doc

    # 2. Score sparse BM25 documents
    for rank, doc in enumerate(sparse_docs):
        key = get_key(doc)
        scores[key] = scores.get(key, 0.0) + (1.0 / (k_const + rank + 1))
        if key not in doc_map:
            doc_map[key] = doc

    # 3. Sort by fused RRF score descending
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    fused_docs: List[Document] = []
    for key in sorted_keys[:top_n]:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = scores[key]
        fused_docs.append(doc)

    return fused_docs
