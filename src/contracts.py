"""Typed contracts for the Turkish Customer Support RAG pipeline."""
from __future__ import annotations

from typing import Literal, TypedDict

CategoryType = Literal["kargo", "iade", "urun", "teknik", "genel"]
IntentType = Literal["sikayet", "bilgi_talebi", "tesekkur"]
RetrievalMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]


class NLUResult(TypedDict):
    category: CategoryType
    category_confidence: float
    intent: IntentType
    intent_confidence: float
    sentiment_score: float
    latency_ms: float
    model_name: str


class RetrievedEvidence(TypedDict):
    policy_id: str
    title: str
    category: str
    content: str
    bm25_rank: int
    dense_rank: int
    bm25_score: float
    dense_score: float
    rrf_score: float
    reranker_score: float | None
    confidence_passed: bool


class RAGResult(TypedDict):
    query: str
    mode: RetrievalMode
    dense_backend: str
    reranker_active: bool
    top_evidences: list[RetrievedEvidence]
    best_policy_id: str
    grounding_verified: bool
    retrieval_latency_ms: float
    rerank_latency_ms: float


class GenerationResult(TypedDict):
    reply: str
    generation_mode: str
    numeric_grounding_passed: bool
    attempts: list[dict[str, object]]
    latency_ms: float


class ChatResponse(TypedDict):
    reply: str
    policy: dict[str, str]
    evidences: list[RetrievedEvidence]
    nlu: NLUResult
    grounding: dict[str, object]
    timings_ms: dict[str, float]
    backends: dict[str, object]
