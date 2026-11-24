"""Live PostgreSQL 16 + pgvector integration tests (executed in CI service)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.rag_engine import load_chunks_jsonl
from src.stores.pgvector_store import PgVectorStore

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
ROOT_DIR = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="DATABASE_URL not set; live pgvector integration test runs in Docker/CI.",
)


def _deterministic_vec(text: str, dim: int = 384) -> list[float]:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    vec = rng.normal(size=dim).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    return vec.tolist()


def test_pgvector_schema_upsert_and_hybrid_rrf():
    store = PgVectorStore(DATABASE_URL)
    chunks = load_chunks_jsonl(ROOT_DIR / "data" / "chunks.jsonl")[:40]
    embeddings = [_deterministic_vec(c["content"]) for c in chunks]

    inserted = store.upsert_chunks(chunks, embeddings)
    assert inserted == 40
    assert store.count_chunks() >= 40

    # Lexical Turkish tsvector search
    lex_hits = store.search_lexical("tüketici ayıplı mal", limit=5)
    assert len(lex_hits) >= 1
    assert lex_hits[0]["lexical_rank"] == 1

    # Dense HNSW search
    q_vec = embeddings[0]
    dense_hits = store.search_dense(q_vec, limit=5)
    assert len(dense_hits) >= 1
    assert dense_hits[0]["chunk"]["chunk_id"] == chunks[0]["chunk_id"]
    assert dense_hits[0]["dense_score"] > 0.99

    # SQL CTE Hybrid RRF search with metadata filter
    hyb_hits = store.search_hybrid(
        "tüketici kanunu",
        q_vec,
        limit=5,
        filters={"doc_id": chunks[0]["doc_id"]},
    )
    assert len(hyb_hits) >= 1
    assert all(h["chunk"]["doc_id"] == chunks[0]["doc_id"] for h in hyb_hits)

    h = store.health()
    assert h["backend"] == "pgvector"
    assert h["dense_indexed"] is True
