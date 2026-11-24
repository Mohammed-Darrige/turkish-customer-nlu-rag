"""Typed contracts for the Turkish Customer Support RAG v2 pipeline."""
from __future__ import annotations

from typing import Literal, TypedDict

CategoryType = Literal["kargo", "iade", "urun", "teknik", "genel"]
IntentType = Literal["sikayet", "bilgi_talebi", "tesekkur"]
LangType = Literal["tr", "ar", "en"]
RetrievalMode = Literal["lexical", "bm25", "dense", "hybrid", "hybrid_rerank"]


class DocumentChunk(TypedDict):
    chunk_id: str
    doc_id: str
    title: str
    article_no: str
    citation_tag: str
    category: CategoryType
    lang: str
    effective_date: str
    source_url: str
    content: str
    token_count: int


class RetrievalFilter(TypedDict, total=False):
    doc_id: str | None
    category: CategoryType | None
    lang: str | None


class NLUResult(TypedDict):
    language: LangType
    category: CategoryType
    category_confidence: float
    intent: IntentType
    intent_confidence: float
    sentiment_score: float
    latency_ms: float
    model_name: str


class RetrievedEvidence(TypedDict):
    chunk_id: str
    doc_id: str
    title: str
    article_no: str
    citation_tag: str
    category: str
    lang: str
    source_url: str
    content: str
    lexical_rank: int
    dense_rank: int
    lexical_score: float
    dense_score: float
    rrf_score: float
    reranker_score: float | None
    confidence_passed: bool


class RAGResult(TypedDict):
    query: str
    mode: RetrievalMode
    store_backend: str
    dense_backend: str
    reranker_active: bool
    abstained: bool
    abstention_reason: str | None
    top_evidences: list[RetrievedEvidence]
    best_chunk_id: str | None
    best_citation_tag: str | None
    grounding_verified: bool
    retrieval_latency_ms: float
    rerank_latency_ms: float


class GenerationResult(TypedDict):
    reply: str
    generation_mode: str
    numeric_grounding_passed: bool
    citation_grounding_passed: bool
    ungrounded_citations: list[str]
    attempts: list[dict[str, object]]
    latency_ms: float


class ChatResponse(TypedDict):
    reply: str
    abstained: bool
    abstention_reason: str | None
    policy: dict[str, str]
    evidences: list[RetrievedEvidence]
    nlu: NLUResult
    grounding: dict[str, object]
    timings_ms: dict[str, float]
    backends: dict[str, object]
