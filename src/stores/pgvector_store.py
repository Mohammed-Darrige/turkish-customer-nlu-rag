"""
PostgreSQL 16 + pgvector RetrieverStore (HNSW + Turkish tsvector GIN + SQL CTE RRF).

Executes lexical (tsvector + ts_rank_cd), dense (HNSW cosine distance `<=>`),
and SQL-native Hybrid Reciprocal Rank Fusion (CTE + FULL OUTER JOIN) directly
inside PostgreSQL 16.
"""
from __future__ import annotations

import os
from typing import Any

from src.contracts import DocumentChunk, RetrievalFilter
from src.stores.base import StoreHit

EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "384"))

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    title TEXT NOT NULL,
    article_no TEXT NOT NULL,
    citation_tag TEXT NOT NULL,
    category TEXT NOT NULL,
    lang TEXT NOT NULL,
    effective_date DATE NOT NULL,
    source_url TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INT NOT NULL,
    content_tsvector tsvector,
    embedding vector({EMBEDDING_DIM})
);

CREATE OR REPLACE FUNCTION chunks_tsvector_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsvector :=
        setweight(to_tsvector('turkish', coalesce(NEW.title, '') || ' ' || coalesce(NEW.article_no, '')), 'A') ||
        setweight(to_tsvector('turkish', coalesce(NEW.content, '')), 'B');
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_chunks_tsvector ON chunks;
CREATE TRIGGER trg_chunks_tsvector
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_tsvector_trigger();

CREATE INDEX IF NOT EXISTS idx_chunks_tsvector
    ON chunks USING gin (content_tsvector);

CREATE INDEX IF NOT EXISTS idx_chunks_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_filters
    ON chunks (doc_id, category, lang);
"""


def _build_where(filters: RetrievalFilter | None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if filters:
        if filters.get("doc_id"):
            clauses.append("doc_id = %s")
            params.append(filters["doc_id"])
        if filters.get("category"):
            clauses.append("category = %s")
            params.append(filters["category"])
        if filters.get("lang"):
            clauses.append("lang = %s")
            params.append(filters["lang"])
    where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where_sql, params


def _row_to_chunk(row: tuple[Any, ...]) -> DocumentChunk:
    return {
        "chunk_id": str(row[0]),
        "doc_id": str(row[1]),
        "title": str(row[2]),
        "article_no": str(row[3]),
        "citation_tag": str(row[4]),
        "category": row[5],
        "lang": str(row[6]),
        "effective_date": str(row[7]),
        "source_url": str(row[8]),
        "content": str(row[9]),
        "token_count": int(row[10]),
    }


class PgVectorStore:
    """PostgreSQL 16 + pgvector HNSW + Turkish tsvector Hybrid RRF store."""

    backend_name = "pgvector"

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._ensure_schema()

    def _connect(self):
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self.database_url)
        register_vector(conn)
        return conn

    def _ensure_schema(self) -> None:
        import psycopg

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
            conn.commit()

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]] | None = None,
    ) -> int:
        if not chunks:
            return 0
        sql = """
            INSERT INTO chunks (
                chunk_id, doc_id, title, article_no, citation_tag,
                category, lang, effective_date, source_url, content,
                token_count, embedding
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                title = EXCLUDED.title,
                article_no = EXCLUDED.article_no,
                citation_tag = EXCLUDED.citation_tag,
                category = EXCLUDED.category,
                lang = EXCLUDED.lang,
                effective_date = EXCLUDED.effective_date,
                source_url = EXCLUDED.source_url,
                content = EXCLUDED.content,
                token_count = EXCLUDED.token_count,
                embedding = COALESCE(EXCLUDED.embedding, chunks.embedding)
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                for idx, ch in enumerate(chunks):
                    emb = embeddings[idx] if embeddings and idx < len(embeddings) else None
                    cur.execute(
                        sql,
                        (
                            ch["chunk_id"],
                            ch["doc_id"],
                            ch["title"],
                            ch["article_no"],
                            ch["citation_tag"],
                            ch["category"],
                            ch["lang"],
                            ch["effective_date"],
                            ch["source_url"],
                            ch["content"],
                            ch["token_count"],
                            emb,
                        ),
                    )
            conn.commit()
        return len(chunks)

    def search_lexical(
        self,
        query: str,
        limit: int = 20,
        filters: RetrievalFilter | None = None,
    ) -> list[StoreHit]:
        where_sql, f_params = _build_where(filters)
        sql = f"""
            SELECT
                chunk_id, doc_id, title, article_no, citation_tag,
                category, lang, effective_date::text, source_url, content, token_count,
                ts_rank_cd(content_tsvector, websearch_to_tsquery('turkish', %s)) AS lex_score
            FROM chunks
            {where_sql}
            ORDER BY lex_score DESC, chunk_id ASC
            LIMIT %s
        """
        params: list[Any] = [query, *f_params, limit]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        hits: list[StoreHit] = []
        for rank, row in enumerate(rows, start=1):
            hits.append({
                "chunk": _row_to_chunk(row),
                "lexical_rank": rank,
                "dense_rank": rank,
                "lexical_score": float(row[11] or 0.0),
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
        where_sql, f_params = _build_where(filters)
        extra_cond = "embedding IS NOT NULL"
        full_where = f"{where_sql} AND {extra_cond}" if where_sql else f"WHERE {extra_cond}"
        sql = f"""
            SELECT
                chunk_id, doc_id, title, article_no, citation_tag,
                category, lang, effective_date::text, source_url, content, token_count,
                (1.0 - (embedding <=> %s::vector)) AS dense_score
            FROM chunks
            {full_where}
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s
        """
        params: list[Any] = [query_embedding, *f_params, query_embedding, limit]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        hits: list[StoreHit] = []
        for rank, row in enumerate(rows, start=1):
            hits.append({
                "chunk": _row_to_chunk(row),
                "lexical_rank": rank,
                "dense_rank": rank,
                "lexical_score": 0.0,
                "dense_score": float(row[11] or 0.0),
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
        if query_embedding is None:
            return self.search_lexical(query, limit=limit, filters=filters)

        where_sql, f_params = _build_where(filters)
        dense_where = f"{where_sql} AND embedding IS NOT NULL" if where_sql else "WHERE embedding IS NOT NULL"
        pool_limit = max(limit * 3, 40)

        sql = f"""
            WITH lexical AS (
                SELECT
                    chunk_id,
                    ts_rank_cd(content_tsvector, websearch_to_tsquery('turkish', %s)) AS lex_score,
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank_cd(content_tsvector, websearch_to_tsquery('turkish', %s)) DESC,
                                 chunk_id ASC
                    ) AS lex_rank
                FROM chunks
                {where_sql}
                LIMIT %s
            ),
            semantic AS (
                SELECT
                    chunk_id,
                    (1.0 - (embedding <=> %s::vector)) AS den_score,
                    ROW_NUMBER() OVER (
                        ORDER BY embedding <=> %s::vector ASC
                    ) AS den_rank
                FROM chunks
                {dense_where}
                LIMIT %s
            ),
            fused AS (
                SELECT
                    COALESCE(l.chunk_id, s.chunk_id) AS chunk_id,
                    COALESCE(l.lex_rank, %s) AS lex_rank,
                    COALESCE(s.den_rank, %s) AS den_rank,
                    COALESCE(l.lex_score, 0.0) AS lex_score,
                    COALESCE(s.den_score, 0.0) AS den_score,
                    (COALESCE(1.0 / (%s + l.lex_rank), 0.0) +
                     COALESCE(1.0 / (%s + s.den_rank), 0.0)) AS rrf_score
                FROM lexical l
                FULL OUTER JOIN semantic s ON l.chunk_id = s.chunk_id
            )
            SELECT
                c.chunk_id, c.doc_id, c.title, c.article_no, c.citation_tag,
                c.category, c.lang, c.effective_date::text, c.source_url, c.content, c.token_count,
                f.lex_rank, f.den_rank, f.lex_score, f.den_score, f.rrf_score
            FROM fused f
            JOIN chunks c ON c.chunk_id = f.chunk_id
            ORDER BY f.rrf_score DESC, f.den_score DESC
            LIMIT %s
        """
        fallback_rank = pool_limit + 1
        params: list[Any] = [
            query, query, *f_params, pool_limit,
            query_embedding, query_embedding, *f_params, pool_limit,
            fallback_rank, fallback_rank, rrf_k, rrf_k,
            limit,
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        hits: list[StoreHit] = []
        for row in rows:
            hits.append({
                "chunk": _row_to_chunk(row),
                "lexical_rank": int(row[11]),
                "dense_rank": int(row[12]),
                "lexical_score": float(row[13] or 0.0),
                "dense_score": float(row[14] or 0.0),
                "rrf_score": float(row[15] or 0.0),
            })
        return hits

    def count_chunks(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks")
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def list_chunks(self, limit: int = 500) -> list[DocumentChunk]:
        sql = """
            SELECT
                chunk_id, doc_id, title, article_no, citation_tag,
                category, lang, effective_date::text, source_url, content, token_count
            FROM chunks
            ORDER BY doc_id ASC, chunk_id ASC
            LIMIT %s
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall()
        return [_row_to_chunk(r) for r in rows]

    def health(self) -> dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*), COUNT(embedding) FROM chunks")
                row = cur.fetchone()
                total = int(row[0]) if row else 0
                embedded = int(row[1]) if row else 0
        return {
            "backend": self.backend_name,
            "chunk_count": total,
            "dense_indexed": embedded > 0,
            "embedded_chunks": embedded,
        }
