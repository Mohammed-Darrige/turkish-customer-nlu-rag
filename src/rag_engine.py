"""
Hybrid Turkish & Multilingual RAG Engine v2.

Orchestrates a pluggable `RetrieverStore` (`PgVectorStore` or `InMemoryStore`),
dense embeddings (`intfloat/multilingual-e5-small`), optional Cross-Encoder
reranking, and a calibrated abstention gate. Contains zero hardcoded keyword
bonuses (`kw_bonus` removed).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.contracts import (
    DocumentChunk,
    RAGResult,
    RetrievalFilter,
    RetrievalMode,
    RetrievedEvidence,
)
from src.stores.base import RetrieverStore, StoreHit
from src.stores.memory_store import InMemoryStore, tokenize_text
from src.stores.pgvector_store import PgVectorStore

os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

ROOT_DIR = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT_DIR / "data" / "chunks.jsonl"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
RRF_K = 60
CE_WEIGHT = 0.5
DEFAULT_ABSTAIN_DENSE_THRESHOLD = float(os.getenv("RAG_ABSTAIN_DENSE_THRESHOLD", "0.845"))
DEFAULT_ABSTAIN_LEXICAL_THRESHOLD = float(os.getenv("RAG_ABSTAIN_LEXICAL_THRESHOLD", "2.0"))


def load_chunks_jsonl(path: Path = CHUNKS_PATH) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                chunks.append(json.loads(s))
    return chunks


class EmbeddingEncoder:
    """Lazy wrapper around sentence-transformers for query and passage encoding."""

    def __init__(self) -> None:
        self.model: Any = None
        self.backend: str = "disabled"
        self._init_model()

    def _init_model(self) -> None:
        if os.getenv("RAG_ENABLE_DENSE", "1") == "0":
            self.backend = "disabled"
            return

        model_name = os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.backend = f"sentence_transformers:{model_name}"
        except Exception:
            self.model = None
            self.backend = "unavailable"

    @property
    def is_active(self) -> bool:
        return self.model is not None

    def encode_passages(self, chunks: list[DocumentChunk]) -> list[list[float]] | None:
        if self.model is None or not chunks:
            return None
        passages = [f"passage: {c['title']} - {c['content']}" for c in chunks]
        embs = self.model.encode(passages, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embs, dtype=np.float32).tolist()

    def encode_query(self, query: str) -> list[float] | None:
        if self.model is None:
            return None
        emb = self.model.encode([f"query: {query}"], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(emb, dtype=np.float32).tolist()


def _create_store() -> RetrieverStore:
    backend = os.getenv("RAG_STORE_BACKEND", "auto").strip().lower()
    db_url = os.getenv("DATABASE_URL", "").strip()
    if backend == "pgvector" or (backend == "auto" and db_url):
        if not db_url:
            raise ValueError("RAG_STORE_BACKEND=pgvector requires DATABASE_URL to be set.")
        return PgVectorStore(db_url)
    return InMemoryStore()


class HybridRAGEngine:
    """Orchestrates RetrieverStore, Dense E5, CrossEncoder reranker, and abstention."""

    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        store: RetrieverStore | None = None,
    ) -> None:
        self.chunks: list[DocumentChunk] = load_chunks_jsonl(chunks_path)
        self.encoder = EmbeddingEncoder()
        self.store: RetrieverStore = store if store is not None else _create_store()

        # Index chunks into the store if empty (or always for InMemoryStore)
        if self.store.backend_name == "in_memory" or self.store.count_chunks() < len(self.chunks):
            embeddings = self.encoder.encode_passages(self.chunks)
            self.store.upsert_chunks(self.chunks, embeddings)

        self.reranker_model: Any = None
        self.reranker_name: str | None = None
        self._init_reranker()

    def _init_reranker(self) -> None:
        reranker_id = os.getenv("RAG_RERANKER_MODEL", "").strip()
        if not reranker_id:
            return
        try:
            from sentence_transformers import CrossEncoder

            self.reranker_model = CrossEncoder(reranker_id)
            self.reranker_name = reranker_id
        except Exception:
            self.reranker_model = None
            self.reranker_name = None

    def _evaluate_abstention(
        self,
        query: str,
        top_hit: StoreHit | None,
        mode: RetrievalMode,
    ) -> tuple[bool, str | None]:
        if top_hit is None:
            return True, "no_candidates"

        q_tokens = set(tokenize_text(query))
        c = top_hit["chunk"]
        doc_tokens = set(tokenize_text(f"{c['title']} {c['content']}"))
        overlap = len(q_tokens & doc_tokens)

        dense_active = self.encoder.is_active and mode != "lexical" and mode != "bm25"
        if dense_active:
            if top_hit["dense_score"] < DEFAULT_ABSTAIN_DENSE_THRESHOLD:
                return (
                    True,
                    f"low_relevance(dense={top_hit['dense_score']:.3f}<{DEFAULT_ABSTAIN_DENSE_THRESHOLD})",
                )
            return False, None

        # Lexical-only abstention check
        if top_hit["lexical_score"] < DEFAULT_ABSTAIN_LEXICAL_THRESHOLD and overlap < 2:
            return (
                True,
                f"low_lexical_relevance(score={top_hit['lexical_score']:.3f},overlap={overlap})",
            )
        return False, None

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        mode: RetrievalMode = "hybrid",
        filters: RetrievalFilter | None = None,
    ) -> RAGResult:
        t0 = time.perf_counter()
        normalized_mode: RetrievalMode = "lexical" if mode == "bm25" else mode
        pool_k = max(top_k * 4, 20)

        query_emb = None
        if normalized_mode != "lexical" and self.encoder.is_active:
            query_emb = self.encoder.encode_query(query)

        if normalized_mode == "lexical":
            hits = self.store.search_lexical(query, limit=pool_k, filters=filters)
        elif normalized_mode == "dense" and query_emb is not None:
            hits = self.store.search_dense(query_emb, limit=pool_k, filters=filters)
        else:
            hits = self.store.search_hybrid(
                query, query_emb, limit=pool_k, rrf_k=RRF_K, filters=filters
            )

        t_retrieval = (time.perf_counter() - t0) * 1000

        rerank_scores: dict[str, float] = {}
        t_rerank_start = time.perf_counter()
        if normalized_mode == "hybrid_rerank" and self.reranker_model is not None and hits:
            pairs = [[query, f"{h['chunk']['title']}: {h['chunk']['content']}"] for h in hits]
            ce_raw = np.asarray(self.reranker_model.predict(pairs), dtype=float)
            ce_z = (ce_raw - ce_raw.mean()) / (ce_raw.std() + 1e-6)
            scored: list[tuple[float, StoreHit, float]] = []
            for h, raw_s, z_s in zip(hits, ce_raw, ce_z):
                combined = h["rrf_score"] * 10.0 + CE_WEIGHT * float(z_s)
                scored.append((combined, h, float(raw_s)))
            scored.sort(key=lambda x: x[0], reverse=True)
            hits = [item[1] for item in scored]
            rerank_scores = {item[1]["chunk"]["chunk_id"]: round(item[2], 4) for item in scored}
        t_rerank = (time.perf_counter() - t_rerank_start) * 1000

        top_hit = hits[0] if hits else None
        abstained, abstention_reason = self._evaluate_abstention(query, top_hit, normalized_mode)

        q_tokens = set(tokenize_text(query))
        evidences: list[RetrievedEvidence] = []
        for h in hits[:top_k]:
            ch = h["chunk"]
            doc_tokens = set(tokenize_text(f"{ch['title']} {ch['content']}"))
            overlap = len(q_tokens & doc_tokens)
            passed = not abstained and (
                overlap >= 1
                or (self.encoder.is_active and h["dense_score"] >= DEFAULT_ABSTAIN_DENSE_THRESHOLD)
            )
            evidences.append({
                "chunk_id": ch["chunk_id"],
                "doc_id": ch["doc_id"],
                "title": ch["title"],
                "article_no": ch["article_no"],
                "citation_tag": ch["citation_tag"],
                "category": ch["category"],
                "lang": ch["lang"],
                "source_url": ch["source_url"],
                "content": ch["content"],
                "lexical_rank": h["lexical_rank"],
                "dense_rank": h["dense_rank"],
                "lexical_score": round(h["lexical_score"], 4),
                "dense_score": round(h["dense_score"], 4),
                "rrf_score": round(h["rrf_score"], 5),
                "reranker_score": rerank_scores.get(ch["chunk_id"]),
                "confidence_passed": passed,
            })

        best_ev = evidences[0] if (evidences and not abstained) else None
        return {
            "query": query,
            "mode": mode,
            "store_backend": self.store.backend_name,
            "dense_backend": self.encoder.backend,
            "reranker_active": self.reranker_model is not None,
            "abstained": abstained,
            "abstention_reason": abstention_reason,
            "top_evidences": evidences,
            "best_chunk_id": best_ev["chunk_id"] if best_ev else None,
            "best_citation_tag": best_ev["citation_tag"] if best_ev else None,
            "grounding_verified": bool(best_ev and best_ev["confidence_passed"]),
            "retrieval_latency_ms": round(t_retrieval, 2),
            "rerank_latency_ms": round(t_rerank, 2),
        }


_global_engine: HybridRAGEngine | None = None


def get_rag_engine() -> HybridRAGEngine:
    global _global_engine
    if _global_engine is None:
        _global_engine = HybridRAGEngine()
    return _global_engine


def retrieve_policies(
    query: str,
    top_k: int = 3,
    mode: RetrievalMode = "hybrid",
    filters: RetrievalFilter | None = None,
) -> RAGResult:
    return get_rag_engine().retrieve(query, top_k=top_k, mode=mode, filters=filters)
