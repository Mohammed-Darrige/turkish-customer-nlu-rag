"""Abstract seam for retrieval backends (InMemoryStore and PgVectorStore)."""
from __future__ import annotations

from typing import Any, Protocol, TypedDict

from src.contracts import DocumentChunk, RetrievalFilter


class StoreHit(TypedDict):
    chunk: DocumentChunk
    lexical_rank: int
    dense_rank: int
    lexical_score: float
    dense_score: float
    rrf_score: float


class RetrieverStore(Protocol):
    """Deep storage seam supporting lexical, dense, and RRF hybrid search."""

    backend_name: str

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> int:
        ...

    def search_lexical(
        self,
        query: str,
        limit: int = 20,
        filters: RetrievalFilter | None = None,
    ) -> list[StoreHit]:
        ...

    def search_dense(
        self,
        query_embedding: list[float],
        limit: int = 20,
        filters: RetrievalFilter | None = None,
    ) -> list[StoreHit]:
        ...

    def search_hybrid(
        self,
        query: str,
        query_embedding: list[float] | None,
        limit: int = 20,
        rrf_k: int = 60,
        filters: RetrievalFilter | None = None,
    ) -> list[StoreHit]:
        ...

    def count_chunks(self) -> int:
        ...

    def list_chunks(self, limit: int = 500) -> list[DocumentChunk]:
        ...

    def health(self) -> dict[str, Any]:
        ...
