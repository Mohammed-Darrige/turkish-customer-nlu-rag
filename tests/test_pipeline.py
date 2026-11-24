"""Unit and API integration tests for Turkish Customer Support RAG v2."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from src.generator import (
    extract_citations,
    generate_reply,
    ungrounded_citations,
    ungrounded_numbers,
)
from src.ingestion.chunker import load_store_ops_chunks
from src.nlu import classify_text, detect_language
from src.rag_engine import HybridRAGEngine, load_chunks_jsonl
from src.server import app

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_corpus_chunks_and_provenance():
    chunks = load_chunks_jsonl(ROOT_DIR / "data" / "chunks.jsonl")
    assert len(chunks) >= 200
    doc_ids = {c["doc_id"] for c in chunks}
    assert {"kanun_6502", "mesafeli_sozlesmeler", "garanti_belgesi", "satis_sonrasi", "store_ops_faq"}.issubset(doc_ids)
    for c in chunks:
        assert c["content"].strip()
        assert c["citation_tag"].startswith(f"{c['doc_id']}:")
        assert c["source_url"]


def test_nlu_language_and_intent():
    tr_res = classify_text("Kargom 5 gündür gelmedi, nerede?")
    assert tr_res["language"] == "tr"
    assert tr_res["category"] == "kargo"
    assert tr_res["intent"] == "sikayet"

    ar_res = classify_text("كيف أستخرج كود الإرجاع المجاني؟")
    assert ar_res["language"] == "ar"

    en_res = classify_text("How do I get a free return shipping code?")
    assert en_res["language"] == "en"
    assert detect_language("Sipariş iadesi") == "tr"


def test_bm25_and_metadata_filtering(monkeypatch):
    monkeypatch.setenv("RAG_ENABLE_DENSE", "0")
    engine = HybridRAGEngine()
    res = engine.retrieve("14 gün içinde cayma hakkı", top_k=3, mode="lexical")
    assert res["store_backend"] == "in_memory"
    assert len(res["top_evidences"]) == 3
    top_tags = [ev["citation_tag"] for ev in res["top_evidences"]]
    assert any("mesafeli_sozlesmeler" in t or "kanun_6502" in t for t in top_tags)

    filtered = engine.retrieve(
        "kargo takip",
        top_k=3,
        mode="lexical",
        filters={"doc_id": "store_ops_faq", "lang": "tr"},
    )
    assert all(ev["doc_id"] == "store_ops_faq" and ev["lang"] == "tr" for ev in filtered["top_evidences"])


def test_numeric_and_citation_grounding():
    assert ungrounded_numbers("14 gün içinde iade edebilirsiniz.", "14 gün içinde cayma hakkı.") == []
    assert ungrounded_numbers("30 gün içinde iade edebilirsiniz.", "14 gün içinde cayma hakkı.") == ["30"]

    sample_ev = [{
        "chunk_id": "mesafeli_sozlesmeler__madde_9__c0",
        "doc_id": "mesafeli_sozlesmeler",
        "title": "Mesafeli Sözleşmeler Yönetmeliği | Madde 9",
        "article_no": "Madde 9",
        "citation_tag": "mesafeli_sozlesmeler:Madde 9",
        "category": "iade",
        "lang": "tr",
        "source_url": "https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=20237&MevzuatTur=7&MevzuatTertip=5",
        "content": "Tüketici, on dört gün içinde herhangi bir gerekçe göstermeksizin sözleşmeden cayma hakkına sahiptir.",
        "lexical_rank": 1,
        "dense_rank": 1,
        "lexical_score": 5.0,
        "dense_score": 0.9,
        "rrf_score": 0.032,
        "reranker_score": None,
        "confidence_passed": True,
    }]
    assert extract_citations("14 gün içinde cayabilirsiniz [mesafeli_sozlesmeler:Madde 9].") == ["mesafeli_sozlesmeler:Madde 9"]
    assert ungrounded_citations("14 gün [mesafeli_sozlesmeler:Madde 9].", sample_ev) == []
    assert ungrounded_citations("14 gün [kanun_6502:Madde 999].", sample_ev) == ["kanun_6502:Madde 999"]
    assert ungrounded_citations("Etiketsiz yanıt.", sample_ev) == ["MISSING_CITATION"]

    gen = generate_reply(
        message="14 gün iade",
        evidences=sample_ev,
        nlu=classify_text("14 gün iade"),
        grounding_verified=True,
        abstained=False,
        use_llm=False,
    )
    assert gen["citation_grounding_passed"] is True
    assert "[mesafeli_sozlesmeler:Madde 9]" in gen["reply"]


def test_api_chat_and_abstention(monkeypatch):
    monkeypatch.setenv("RAG_ENABLE_DENSE", "0")
    with TestClient(app) as client:
        h = client.get("/api/health")
        assert h.status_code == 200
        assert h.json()["chunk_count"] >= 200
        assert h.json()["store_backend"] == "in_memory"

        c = client.get("/api/chunks?limit=5")
        assert c.status_code == 200
        assert len(c.json()["chunks"]) == 5

        ok = client.post(
            "/api/chat",
            json={
                "message": "14 gün içinde cayma hakkı nasıl kullanılır?",
                "retrieval_mode": "lexical",
                "use_llm": False,
            },
        )
        assert ok.status_code == 200
        body = ok.json()
        assert body["abstained"] is False
        assert body["grounding"]["citation_grounding_passed"] is True
        assert body["policy"]["citation_tag"]

        trap = client.post(
            "/api/chat",
            json={
                "message": "xzqvw plmokn qazwsx",
                "retrieval_mode": "lexical",
                "use_llm": False,
            },
        )
        assert trap.status_code == 200
        assert trap.json()["abstained"] is True
        assert trap.json()["backends"]["generation"] == "abstention_guard"
