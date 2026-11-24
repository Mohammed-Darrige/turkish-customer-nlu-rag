"""
FastAPI server for the Turkish Customer Support Voice & Text RAG v2 Assistant.

Run locally:
    uvicorn src.server:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import generator, nlu, rag_engine
from src.contracts import CategoryType, RetrievalFilter

logger = logging.getLogger("turkish_rag")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"

_state = {
    "ready": False,
    "nlu_backend": "uninitialized",
    "store_backend": "uninitialized",
    "dense_backend": "uninitialized",
    "reranker_active": False,
    "chunk_count": 0,
}

_metrics = {
    "requests_total": 0,
    "chat_requests_total": 0,
    "abstentions_total": 0,
    "citation_guard_failures_total": 0,
    "retrieval_ms_sum": 0.0,
    "rerank_ms_sum": 0.0,
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine = rag_engine.get_rag_engine()
    _state["nlu_backend"] = nlu.get_nlu_backend()
    _state["store_backend"] = engine.store.backend_name
    _state["dense_backend"] = engine.encoder.backend
    _state["reranker_active"] = engine.reranker_model is not None
    _state["chunk_count"] = engine.store.count_chunks()
    _state["ready"] = True
    try:
        yield
    finally:
        if hasattr(engine.store, "close"):
            engine.store.close()


app = FastAPI(
    title="Turkish Customer Support RAG v2 Assistant",
    version="2.5.0",
    docs_url="/api/docs",
    redoc_url=None,
    lifespan=lifespan,
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    _metrics["requests_total"] += 1
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    logger.info(
        json.dumps({
            "event": "http_request",
            "request_id": req_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time_ms": elapsed_ms,
        })
    )
    return response


@app.get("/api/metrics")
def metrics():
    chat_n = max(1, _metrics["chat_requests_total"])
    return {
        "requests_total": _metrics["requests_total"],
        "chat_requests_total": _metrics["chat_requests_total"],
        "abstentions_total": _metrics["abstentions_total"],
        "citation_guard_failures_total": _metrics["citation_guard_failures_total"],
        "avg_retrieval_ms": round(_metrics["retrieval_ms_sum"] / chat_n, 2),
        "avg_rerank_ms": round(_metrics["rerank_ms_sum"] / chat_n, 2),
    }


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(3, ge=1, le=5)
    retrieval_mode: Literal["lexical", "bm25", "dense", "hybrid", "hybrid_rerank"] = "hybrid"
    doc_id: str | None = None
    category: CategoryType | None = None
    lang: str | None = None
    use_llm: bool = True
    model: str | None = None


@app.get("/api/health")
def health():
    engine = rag_engine.get_rag_engine()
    models = generator.installed_models()
    return {
        "ready": _state["ready"] or True,
        "chunk_count": engine.store.count_chunks(),
        "policy_count": engine.store.count_chunks(),
        "store_backend": engine.store.backend_name,
        "nlu_backend": nlu.get_nlu_backend(),
        "rag_dense_backend": engine.encoder.backend,
        "rag_reranker_active": engine.reranker_model is not None,
        "ollama_available": bool(models),
        "ollama_model": generator.pick_model(None, models),
        "ollama_installed": models,
    }


@app.get("/api/chunks")
@app.get("/api/policies")
def list_chunks(limit: int = 500):
    engine = rag_engine.get_rag_engine()
    chunks = engine.store.list_chunks(limit=min(max(limit, 1), 1000))
    return {
        "count": len(chunks),
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "article_no": c["article_no"],
                "citation_tag": c["citation_tag"],
                "category": c["category"],
                "lang": c["lang"],
                "title": c["title"],
                "text": c["content"],
            }
            for c in chunks
        ],
    }


@app.post("/api/chat")
def chat(body: ChatRequest):
    nlu_res = nlu.classify_text(body.message)
    filters: RetrievalFilter = {}
    if body.doc_id:
        filters["doc_id"] = body.doc_id
    if body.category:
        filters["category"] = body.category
    if body.lang:
        filters["lang"] = body.lang

    rag_res = rag_engine.retrieve_policies(
        body.message,
        top_k=body.top_k,
        mode=body.retrieval_mode,
        filters=filters or None,
    )
    top_ev = rag_res["top_evidences"][0] if (rag_res["top_evidences"] and not rag_res["abstained"]) else None

    gen_res = generator.generate_reply(
        message=body.message,
        evidences=rag_res["top_evidences"],
        nlu=nlu_res,
        grounding_verified=rag_res["grounding_verified"],
        abstained=rag_res["abstained"],
        use_llm=body.use_llm,
        requested_model=body.model,
    )

    _metrics["chat_requests_total"] += 1
    if rag_res["abstained"]:
        _metrics["abstentions_total"] += 1
    if not gen_res["citation_grounding_passed"]:
        _metrics["citation_guard_failures_total"] += 1
    _metrics["retrieval_ms_sum"] += rag_res["retrieval_latency_ms"]
    _metrics["rerank_ms_sum"] += rag_res["rerank_latency_ms"]

    return {
        "reply": gen_res["reply"],
        "abstained": rag_res["abstained"],
        "abstention_reason": rag_res["abstention_reason"],
        "policy": {
            "id": top_ev["chunk_id"] if top_ev else "",
            "doc_id": top_ev["doc_id"] if top_ev else "",
            "article_no": top_ev["article_no"] if top_ev else "",
            "citation_tag": top_ev["citation_tag"] if top_ev else "",
            "title": top_ev["title"] if top_ev else "",
            "category": top_ev["category"] if top_ev else "",
            "text": top_ev["content"] if top_ev else "",
        },
        "evidences": rag_res["top_evidences"],
        "nlu": nlu_res,
        "grounding": {
            "retrieval_verified": rag_res["grounding_verified"],
            "abstained": rag_res["abstained"],
            "numeric_grounding_passed": gen_res["numeric_grounding_passed"],
            "citation_grounding_passed": gen_res["citation_grounding_passed"],
            "ungrounded_citations": gen_res["ungrounded_citations"],
            "attempts": gen_res["attempts"],
        },
        "timings_ms": {
            "nlu": nlu_res["latency_ms"],
            "retrieval": rag_res["retrieval_latency_ms"],
            "rerank": rag_res["rerank_latency_ms"],
            "generation": gen_res["latency_ms"],
        },
        "backends": {
            "store": rag_res["store_backend"],
            "nlu": nlu_res["model_name"],
            "retrieval_mode": rag_res["mode"],
            "dense": rag_res["dense_backend"],
            "reranker_active": rag_res["reranker_active"],
            "generation": gen_res["generation_mode"],
        },
    }


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
