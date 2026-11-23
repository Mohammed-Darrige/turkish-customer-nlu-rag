"""
Hybrid Turkish RAG Engine (Okapi BM25 + Multilingual-E5 Dense + RRF).

Supports explicit retrieval modes ('bm25', 'dense', 'hybrid', 'hybrid_rerank')
for reproducible ablation and transparent backend reporting.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.contracts import RAGResult, RetrievalMode, RetrievedEvidence
from src.nlu import normalize_turkish

os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

ROOT_DIR = Path(__file__).resolve().parents[1]
POLICIES_PATH = ROOT_DIR / "data" / "policies.json"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
RRF_K = 60
CE_WEIGHT = 0.5
DENSE_GROUNDING_THRESHOLD = 0.78

TURKISH_STOPWORDS = {
    "ve", "veya", "ile", "icin", "gibi", "bir", "bu", "su", "o", "ne",
    "nasil", "nerede", "mi", "mu", "miyim", "miyiz", "hala", "cok",
    "daha", "ben", "bana", "benim", "sen", "sana", "biz", "bize", "de", "da",
    "ise", "ki", "ama", "fakat", "lakin", "ancak", "ya", "hem", "her", "hic",
}


def tokenize_turkish(text: str) -> list[str]:
    """Normalizes Turkish text and returns filtered alphanumeric tokens."""
    norm = normalize_turkish(text)
    words = re.findall(r"[a-z0-9]+", norm)
    return [w for w in words if len(w) > 2 and w not in TURKISH_STOPWORDS]


class BM25Retriever:
    """In-memory Okapi BM25 index over the policy corpus."""

    def __init__(self, docs: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.corpus_size = len(docs)
        self.tokenized_docs = [
            tokenize_turkish(f"{d['title']} {d['text']} {' '.join(d.get('keywords', []))}")
            for d in docs
        ]
        self.doc_lens = [len(td) for td in self.tokenized_docs]
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.corpus_size)

        self.df: dict[str, int] = {}
        for td in self.tokenized_docs:
            for term in set(td):
                self.df[term] = self.df.get(term, 0) + 1

        self.idf: dict[str, float] = {
            term: math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }

    def score(self, query: str) -> list[float]:
        q_tokens = tokenize_turkish(query)
        scores: list[float] = []
        for doc_tokens, doc_len in zip(self.tokenized_docs, self.doc_lens):
            score = 0.0
            doc_counts: dict[str, int] = {}
            for t in doc_tokens:
                doc_counts[t] = doc_counts.get(t, 0) + 1
            for qt in q_tokens:
                if qt in doc_counts:
                    tf = doc_counts[qt]
                    idf = self.idf.get(qt, 0.0)
                    num = tf * (self.k1 + 1.0)
                    den = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += idf * (num / den)
            scores.append(score)
        return scores


class DenseRetriever:
    """Semantic vector retriever using sentence-transformers (multilingual-e5-small)."""

    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs
        self.model: Any = None
        self.doc_embeddings: np.ndarray | None = None
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
            passages = [f"passage: {d['title']} - {d['text']}" for d in self.docs]
            self.doc_embeddings = self.model.encode(
                passages, normalize_embeddings=True, show_progress_bar=False
            )
            self.backend = f"sentence_transformers:{model_name}"
        except Exception:
            self.model = None
            self.doc_embeddings = None
            self.backend = "jaccard_fallback"

    def score(self, query: str) -> list[float]:
        if self.model is not None and self.doc_embeddings is not None:
            q_emb = self.model.encode(
                [f"query: {query}"], normalize_embeddings=True, show_progress_bar=False
            )[0]
            return np.dot(self.doc_embeddings, q_emb).tolist()

        if self.backend == "disabled":
            return [0.0 for _ in self.docs]

        # Transparent Jaccard overlap fallback only when dense was requested but failed to load
        q_tokens = set(tokenize_turkish(query))
        scores: list[float] = []
        for d in self.docs:
            d_tokens = set(tokenize_turkish(f"{d['title']} {d['text']}"))
            union = len(q_tokens | d_tokens)
            scores.append(float(len(q_tokens & d_tokens) / union) if union else 0.0)
        return scores


class HybridRAGEngine:
    """Orchestrates BM25, Dense E5, RRF fusion, and optional Cross-Encoder reranking."""

    def __init__(self, policies_path: Path = POLICIES_PATH):
        with open(policies_path, "r", encoding="utf-8") as f:
            self.policies: list[dict[str, Any]] = json.load(f)

        self.bm25 = BM25Retriever(self.policies)
        self.dense = DenseRetriever(self.policies)
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

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        mode: RetrievalMode = "hybrid",
    ) -> RAGResult:
        t0 = time.perf_counter()

        bm25_scores = self.bm25.score(query)
        bm25_ranked = np.argsort(bm25_scores)[::-1]
        bm25_rank_dict = {int(idx): rank for rank, idx in enumerate(bm25_ranked)}

        dense_active = self.dense.backend.startswith("sentence_transformers:")
        if mode == "bm25" or self.dense.backend == "disabled":
            dense_scores = [0.0] * len(self.policies)
            dense_rank_dict = bm25_rank_dict
        else:
            dense_scores = self.dense.score(query)
            dense_ranked = np.argsort(dense_scores)[::-1]
            dense_rank_dict = {int(idx): rank for rank, idx in enumerate(dense_ranked)}

        t_retrieval = (time.perf_counter() - t0) * 1000

        q_norm = normalize_turkish(query)
        candidates: list[dict[str, Any]] = []
        for idx, pol in enumerate(self.policies):
            r_bm25 = bm25_rank_dict[idx]
            r_dense = dense_rank_dict[idx]
            rrf_score = (1.0 / (RRF_K + r_bm25)) + (1.0 / (RRF_K + r_dense))

            if mode == "bm25":
                primary_score = float(bm25_scores[idx])
            elif mode == "dense":
                primary_score = float(dense_scores[idx])
            else:
                kw_bonus = sum(1.0 for kw in pol.get("keywords", []) if kw in q_norm)
                primary_score = rrf_score * 10.0 + kw_bonus * 0.25 + (float(dense_scores[idx]) * 0.5)

            candidates.append({
                "idx": idx,
                "policy": pol,
                "bm25_rank": r_bm25 + 1,
                "dense_rank": r_dense + 1,
                "bm25_score": float(bm25_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "rrf_score": float(rrf_score),
                "primary_score": primary_score,
                "reranker_score": None,
            })

        candidates.sort(key=lambda c: c["primary_score"], reverse=True)

        t_rerank_start = time.perf_counter()
        if mode == "hybrid_rerank" and self.reranker_model is not None:
            pairs = [[query, f"{c['policy']['title']}: {c['policy']['text']}"] for c in candidates]
            ce_scores = np.asarray(self.reranker_model.predict(pairs), dtype=float)
            ce_z = (ce_scores - ce_scores.mean()) / (ce_scores.std() + 1e-6)
            for c, raw_ce, z in zip(candidates, ce_scores, ce_z):
                c["reranker_score"] = round(float(raw_ce), 4)
                c["primary_score"] += CE_WEIGHT * float(z)
            candidates.sort(key=lambda c: c["primary_score"], reverse=True)
        t_rerank = (time.perf_counter() - t_rerank_start) * 1000

        q_tokens = set(tokenize_turkish(query))
        evidences: list[RetrievedEvidence] = []
        for c in candidates[:top_k]:
            pol = c["policy"]
            pol_tokens = set(tokenize_turkish(f"{pol['title']} {pol['text']} {' '.join(pol.get('keywords', []))}"))
            overlap = len(q_tokens & pol_tokens)
            has_confidence = overlap >= 1 or (dense_active and c["dense_score"] >= DENSE_GROUNDING_THRESHOLD)

            evidences.append({
                "policy_id": pol["id"],
                "title": pol["title"],
                "category": pol["category"],
                "content": pol["text"],
                "bm25_rank": c["bm25_rank"],
                "dense_rank": c["dense_rank"],
                "bm25_score": round(c["bm25_score"], 4),
                "dense_score": round(c["dense_score"], 4),
                "rrf_score": round(c["rrf_score"], 5),
                "reranker_score": c["reranker_score"],
                "confidence_passed": has_confidence,
            })

        best_pol_id = evidences[0]["policy_id"] if evidences else "POL_GENEL_01"
        is_grounded = evidences[0]["confidence_passed"] if evidences else False

        return {
            "query": query,
            "mode": mode,
            "dense_backend": self.dense.backend,
            "reranker_active": self.reranker_model is not None,
            "top_evidences": evidences,
            "best_policy_id": best_pol_id,
            "grounding_verified": is_grounded,
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
) -> RAGResult:
    return get_rag_engine().retrieve(query, top_k=top_k, mode=mode)
