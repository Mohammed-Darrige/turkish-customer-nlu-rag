"""
In-Memory RetrieverStore (Okapi BM25 + Normalized Cosine Dot Product + RRF).

Provides zero-dependency local execution and unit testing behind the exact same
`RetrieverStore` contract as `PgVectorStore`.
"""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from src.contracts import DocumentChunk, RetrievalFilter
from src.nlu import normalize_turkish
from src.stores.base import StoreHit

TURKISH_STOPWORDS = {
    "ve", "veya", "ile", "icin", "gibi", "bir", "bu", "su", "o", "ne",
    "nasil", "nerede", "mi", "mu", "miyim", "miyiz", "hala", "cok",
    "daha", "ben", "bana", "benim", "sen", "sana", "biz", "bize", "de", "da",
    "ise", "ki", "ama", "fakat", "lakin", "ancak", "ya", "hem", "her", "hic",
    "the", "and", "for", "with", "what", "when", "where", "how", "is", "are",
    "of", "in", "to", "an", "on", "can", "do", "does", "my", "your", "its",
}


_TR_STEMMER: Any = None


def _get_tr_stemmer() -> Any:
    global _TR_STEMMER
    if _TR_STEMMER is None:
        try:
            import snowballstemmer

            _TR_STEMMER = snowballstemmer.stemmer("turkish")
        except Exception:
            _TR_STEMMER = False
    return _TR_STEMMER if _TR_STEMMER is not False else None


def tokenize_text(text: str) -> list[str]:
    """Normalizes Turkish/Latin text, applies Snowball Turkish stemming, and filters stopwords."""
    norm = normalize_turkish(text)
    words = [w for w in re.findall(r"[\w]+", norm, flags=re.UNICODE) if len(w) > 1 and w not in TURKISH_STOPWORDS]
    stemmer = _get_tr_stemmer()
    if stemmer is not None and words:
        return stemmer.stemWords(words)
    return words


class InMemoryStore:
    """In-memory Okapi BM25 + dense cosine store implementing `RetrieverStore`."""

    backend_name = "in_memory"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[DocumentChunk] = []
        self._embeddings: np.ndarray | None = None
        self._tokenized_docs: list[list[str]] = []
        self._doc_lens: list[int] = []
        self._avg_doc_len: float = 1.0
        self._idf: dict[str, float] = {}

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> int:
        self._chunks = list(chunks)
        self._tokenized_docs = [
            tokenize_text(f"{c['title']} {c['article_no']} {c['content']}")
            for c in self._chunks
        ]
        self._doc_lens = [len(td) for td in self._tokenized_docs]
        n_docs = max(1, len(self._chunks))
        self._avg_doc_len = sum(self._doc_lens) / n_docs

        df: dict[str, int] = {}
        for td in self._tokenized_docs:
            for term in set(td):
                df[term] = df.get(term, 0) + 1
        self._idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

        if embeddings is not None and len(embeddings) == len(self._chunks) and len(embeddings) > 0:
            mat = np.asarray(embeddings, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            self._embeddings = mat / norms
        else:
            self._embeddings = None

        return len(self._chunks)

    def _matches_filter(self, chunk: DocumentChunk, filters: RetrievalFilter | None) -> bool:
        if not filters:
            return True
        if filters.get("doc_id") and chunk["doc_id"] != filters["doc_id"]:
            return False
        if filters.get("category") and chunk["category"] != filters["category"]:
            return False
        if filters.get("lang") and chunk["lang"] != filters["lang"]:
            return False
        return True

    def _bm25_scores(self, query: str) -> list[float]:
        q_tokens = tokenize_text(query)
        scores: list[float] = []
        for doc_tokens, doc_len in zip(self._tokenized_docs, self._doc_lens):
            score = 0.0
            counts: dict[str, int] = {}
            for t in doc_tokens:
                counts[t] = counts.get(t, 0) + 1
            for qt in q_tokens:
                if qt in counts:
                    tf = counts[qt]
                    idf = self._idf.get(qt, 0.0)
                    num = tf * (self.k1 + 1.0)
                    den = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self._avg_doc_len)))
                    score += idf * (num / den)
            scores.append(score)
        return scores

    def _dense_scores(self, query_embedding: list[float] | None) -> list[float]:
        if self._embeddings is None or query_embedding is None or len(self._chunks) == 0:
            return [0.0] * len(self._chunks)
        q_vec = np.asarray(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm > 0.0:
            q_vec = q_vec / q_norm
        return np.dot(self._embeddings, q_vec).tolist()

    def search_lexical(
        self,
        query: str,
        limit: int = 20,
        filters: RetrievalFilter | None = None,
    ) -> list[StoreHit]:
        bm25_scores = self._bm25_scores(query)
        indices = [
            i for i, ch in enumerate(self._chunks)
            if self._matches_filter(ch, filters)
        ]
        indices.sort(key=lambda i: bm25_scores[i], reverse=True)
        hits: list[StoreHit] = []
        for rank, idx in enumerate(indices[:limit], start=1):
            hits.append({
                "chunk": self._chunks[idx],
                "lexical_rank": rank,
                "dense_rank": rank,
                "lexical_score": float(bm25_scores[idx]),
                "dense_score": 0.0,
                "rrf_score": 1.0 / (60 + rank),
            })
        return hits

    def search_dense(
        self,
        query_embedding: list[float],
        limit: int = 20,
        filters: RetrievalFilter | None = None,
    ) -> list[StoreHit]:
        dense_scores = self._dense_scores(query_embedding)
        indices = [
            i for i, ch in enumerate(self._chunks)
            if self._matches_filter(ch, filters)
        ]
        indices.sort(key=lambda i: dense_scores[i], reverse=True)
        hits: list[StoreHit] = []
        for rank, idx in enumerate(indices[:limit], start=1):
            hits.append({
                "chunk": self._chunks[idx],
                "lexical_rank": rank,
                "dense_rank": rank,
                "lexical_score": 0.0,
                "dense_score": float(dense_scores[idx]),
                "rrf_score": 1.0 / (60 + rank),
            })
        return hits

    def search_hybrid(
        self,
        query: str,
        query_embedding: list[float] | None,
        limit: int = 20,
        rrf_k: int = 60,
        filters: RetrievalFilter | None = None,
    ) -> list[StoreHit]:
        bm25_scores = self._bm25_scores(query)
        dense_scores = self._dense_scores(query_embedding)
        has_dense = self._embeddings is not None and query_embedding is not None

        valid_indices = [
            i for i, ch in enumerate(self._chunks)
            if self._matches_filter(ch, filters)
        ]
        if not valid_indices:
            return []

        lex_order = sorted(valid_indices, key=lambda i: bm25_scores[i], reverse=True)
        lex_rank = {idx: r for r, idx in enumerate(lex_order, start=1)}

        if has_dense:
            den_order = sorted(valid_indices, key=lambda i: dense_scores[i], reverse=True)
            den_rank = {idx: r for r, idx in enumerate(den_order, start=1)}
        else:
            den_rank = lex_rank

        hits: list[StoreHit] = []
        for idx in valid_indices:
            rl = lex_rank[idx]
            rd = den_rank[idx]
            rrf = (1.0 / (rrf_k + rl)) + (1.0 / (rrf_k + rd))
            hits.append({
                "chunk": self._chunks[idx],
                "lexical_rank": rl,
                "dense_rank": rd,
                "lexical_score": float(bm25_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "rrf_score": float(rrf),
            })

        hits.sort(key=lambda h: (h["rrf_score"], h["dense_score"], h["lexical_score"]), reverse=True)
        return hits[:limit]

    def count_chunks(self) -> int:
        return len(self._chunks)

    def list_chunks(self, limit: int = 500) -> list[DocumentChunk]:
        return list(self._chunks[:limit])

    def health(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "chunk_count": len(self._chunks),
            "dense_indexed": self._embeddings is not None,
        }
