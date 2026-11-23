"""Behavioral test suite for Turkish Customer Support RAG & Voice Assistant."""
from __future__ import annotations

import os

from fastapi.testclient import TestClient

from src.generator import ungrounded_numbers
from src.nlu import classify_text
from src.rag_engine import HybridRAGEngine
from src.server import app


def test_nlu_standard_and_typo_queries():
    cases = [
        ("Kargom 4 gündür nerede kaldı teslimat yapılmadı", "kargo", "sikayet"),
        ("14 gün cayma hakkı şartları ve iade prosedürü nedir acaba", "iade", "bilgi_talebi"),
        ("Gelen kazağın dikişleri sökük ve ayıplı çıktı", "urun", "sikayet"),
        ("Kartımdan iki kere çekim yapıldı mobil bankacılıkta hata verdi", "teknik", "sikayet"),
        ("Her şey için çok teşekkürler harika bir esnafsınız", "genel", "tesekkur"),
    ]
    for text, exp_cat, exp_int in cases:
        res = classify_text(text)
        assert res["category"] == exp_cat
        assert res["intent"] == exp_int
        assert res["model_name"] == "rule_based_turkish_nlu"

    noisy = classify_text("kargp cok gecikri kurye gelmedi sube nerde")
    assert noisy["category"] == "kargo"
    assert noisy["intent"] == "sikayet"
    assert noisy["sentiment_score"] < 0.0


def test_bm25_and_hybrid_retrieval_and_grounding():
    engine = HybridRAGEngine()
    res_hygiene = engine.retrieve(
        "İç çamaşırının paketini açtım ama denemedim hijyen sebebiyle iade",
        top_k=3,
        mode="bm25",
    )
    assert res_hygiene["best_policy_id"] == "POL_IADE_02"
    assert res_hygiene["grounding_verified"] is True

    res_hybrid = engine.retrieve("Kartımdan iki kere para çekildi", top_k=3, mode="hybrid")
    assert res_hybrid["best_policy_id"] == "POL_TEKNIK_01"
    assert res_hybrid["top_evidences"][0]["rrf_score"] > 0.0

    if os.getenv("RAG_ENABLE_DENSE", "1") == "0":
        off_topic = engine.retrieve("zxqvbnm qwertyu asdfghj", top_k=3, mode="hybrid")
        assert off_topic["grounding_verified"] is False


def test_numeric_grounding_guard():
    policy = "Teslimat süresi 1-3 iş günüdür. İade süresi 14 gündür."
    msg = "Siparişim 5 gündür gelmedi."
    assert ungrounded_numbers("Kargonuz 1-3 iş günü içinde teslim edilir, 5 gündür beklediğinizi görüyoruz.", policy, msg) == []
    assert ungrounded_numbers("Size 250 TL kupon ve 30 gün süre tanımladık.", policy, msg) == ["250", "30"]


def test_fastapi_endpoints():
    with TestClient(app) as client:
        h = client.get("/api/health")
        assert h.status_code == 200
        h_json = h.json()
        assert h_json["ready"] is True
        assert h_json["policy_count"] == 18

        p = client.get("/api/policies")
        assert p.status_code == 200
        assert p.json()["count"] == 18

        bad = client.post("/api/chat", json={"message": ""})
        assert bad.status_code == 422

        chat = client.post(
            "/api/chat",
            json={"message": "14 gün dolmadan cayma hakkımı kullanmak istiyorum", "use_llm": False},
        )
        assert chat.status_code == 200
        data = chat.json()
        assert data["policy"]["id"] == "POL_IADE_01"
        assert data["grounding"]["numeric_grounding_passed"] is True
        assert data["backends"]["generation"] == "policy_template_fallback"
        assert len(data["reply"]) > 10
