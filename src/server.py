"""
FastAPI server for the Turkish Customer Support Voice & Text RAG Assistant.

Run locally:
    uvicorn src.server:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import generator, nlu, rag_engine

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"

_state = {
    "ready": False,
    "nlu_backend": "uninitialized",
    "dense_backend": "uninitialized",
    "reranker_active": False,
    "policy_count": 0,
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine = rag_engine.get_rag_engine()
    _state["nlu_backend"] = nlu.get_nlu_backend()
    _state["dense_backend"] = engine.dense.backend
    _state["reranker_active"] = engine.reranker_model is not None
    _state["policy_count"] = len(engine.policies)
    _state["ready"] = True
    yield


app = FastAPI(
    title="Turkish Customer Support RAG Assistant",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(3, ge=1, le=5)
    retrieval_mode: Literal["bm25", "dense", "hybrid", "hybrid_rerank"] = "hybrid"
    use_llm: bool = True
    model: str | None = None


@app.get("/api/health")
def health():
    engine = rag_engine.get_rag_engine()
    models = generator.installed_models()
    return {
        "ready": _state["ready"] or True,
        "policy_count": len(engine.policies),
        "nlu_backend": nlu.get_nlu_backend(),
        "rag_dense_backend": engine.dense.backend,
        "rag_reranker_active": engine.reranker_model is not None,
        "ollama_available": bool(models),
        "ollama_model": generator.pick_model(None, models),
        "ollama_installed": models,
    }


@app.get("/api/policies")
def list_policies():
    engine = rag_engine.get_rag_engine()
    return {
        "count": len(engine.policies),
        "policies": [
            {"id": p["id"], "category": p["category"], "title": p["title"], "text": p["text"]}
            for p in engine.policies
        ],
    }


@app.post("/api/chat")
def chat(body: ChatRequest):
    nlu_res = nlu.classify_text(body.message)
    rag_res = rag_engine.retrieve_policies(
        body.message,
        top_k=body.top_k,
        mode=body.retrieval_mode,
    )
    top_ev = rag_res["top_evidences"][0] if rag_res["top_evidences"] else None

    gen_res = generator.generate_reply(
        message=body.message,
        evidence=top_ev,
        nlu=nlu_res,
        grounding_verified=rag_res["grounding_verified"],
        use_llm=body.use_llm,
        requested_model=body.model,
    )

    return {
        "reply": gen_res["reply"],
        "policy": {
            "id": top_ev["policy_id"] if top_ev else "",
            "title": top_ev["title"] if top_ev else "",
            "category": top_ev["category"] if top_ev else "",
            "text": top_ev["content"] if top_ev else "",
        },
        "evidences": rag_res["top_evidences"],
        "nlu": nlu_res,
        "grounding": {
            "retrieval_verified": rag_res["grounding_verified"],
            "numeric_grounding_passed": gen_res["numeric_grounding_passed"],
            "attempts": gen_res["attempts"],
        },
        "timings_ms": {
            "nlu": nlu_res["latency_ms"],
            "retrieval": rag_res["retrieval_latency_ms"],
            "rerank": rag_res["rerank_latency_ms"],
            "generation": gen_res["latency_ms"],
        },
        "backends": {
            "nlu": nlu_res["model_name"],
            "retrieval_mode": rag_res["mode"],
            "dense": rag_res["dense_backend"],
            "reranker_active": rag_res["reranker_active"],
            "generation": gen_res["generation_mode"],
        },
    }


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
