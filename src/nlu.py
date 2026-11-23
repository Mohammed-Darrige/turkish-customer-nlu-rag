"""
Turkish NLU Classifier (Category, Intent, Sentiment).

Uses deterministic Turkish keyword/morphological rules by default.
Optionally loads HuggingFace / local sequence classification models when
NLU_CATEGORY_MODEL and NLU_INTENT_MODEL environment variables are explicitly set.
"""
from __future__ import annotations

import os
import time
from typing import Any

from src.contracts import CategoryType, IntentType, NLUResult

_kat_pipe: Any = None
_niyet_pipe: Any = None
_loaded_model_id: str | None = None
_init_attempted = False

KEYWORD_CATEGORY_MAP: dict[str, CategoryType] = {
    "kargo": "kargo",
    "kurye": "kargo",
    "teslimat": "kargo",
    "paket": "kargo",
    "sube": "kargo",
    "iade": "iade",
    "cayma": "iade",
    "para iadesi": "iade",
    "degisim": "iade",
    "geri gonderdim": "iade",
    "beden": "urun",
    "defolu": "urun",
    "kusurlu": "urun",
    "kirik": "urun",
    "bozuk": "urun",
    "garanti": "urun",
    "dikis": "urun",
    "sokuk": "urun",
    "ayipli": "urun",
    "yirtik": "urun",
    "kalite": "urun",
    "odeme": "teknik",
    "iki kere": "teknik",
    "cekim": "teknik",
    "sms": "teknik",
    "sifre": "teknik",
    "kupon": "teknik",
    "uygulama": "teknik",
    "site": "teknik",
    "hesap": "teknik",
    "mesai": "genel",
    "iletisim": "genel",
    "musteri hizmetleri": "genel",
    "telefon": "genel",
}

_CHARMAP = str.maketrans({
    "ç": "c", "Ç": "c",
    "ğ": "g", "Ğ": "g",
    "ı": "i", "I": "i",
    "İ": "i",
    "ö": "o", "Ö": "o",
    "ş": "s", "Ş": "s",
    "ü": "u", "Ü": "u",
})


def normalize_turkish(text: str) -> str:
    """Lowercases and maps Turkish diacritics to ASCII equivalents."""
    return text.lower().translate(_CHARMAP)


def init_nlu_models() -> bool:
    """Optionally initializes transformer pipelines if env vars are explicitly configured."""
    global _kat_pipe, _niyet_pipe, _loaded_model_id, _init_attempted
    if _init_attempted:
        return _kat_pipe is not None and _niyet_pipe is not None

    _init_attempted = True
    cat_model = os.getenv("NLU_CATEGORY_MODEL", "").strip()
    intent_model = os.getenv("NLU_INTENT_MODEL", "").strip()
    if not cat_model or not intent_model:
        return False

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        device = 0 if torch.cuda.is_available() else -1
        tok_kat = AutoTokenizer.from_pretrained(cat_model)
        mod_kat = AutoModelForSequenceClassification.from_pretrained(cat_model)
        _kat_pipe = pipeline("text-classification", model=mod_kat, tokenizer=tok_kat, device=device)

        tok_niyet = AutoTokenizer.from_pretrained(intent_model)
        mod_niyet = AutoModelForSequenceClassification.from_pretrained(intent_model)
        _niyet_pipe = pipeline("text-classification", model=mod_niyet, tokenizer=tok_niyet, device=device)

        _loaded_model_id = f"{cat_model} + {intent_model}"
        return True
    except Exception:
        _kat_pipe = None
        _niyet_pipe = None
        _loaded_model_id = None
        return False


def get_nlu_backend() -> str:
    """Returns the currently active NLU backend name."""
    init_nlu_models()
    return _loaded_model_id if _loaded_model_id else "rule_based_turkish_nlu"


def classify_text(text: str) -> NLUResult:
    """Classifies Turkish customer text into category, intent, and sentiment score."""
    t0 = time.perf_counter()
    init_nlu_models()

    clean_text = text.strip()
    if _kat_pipe is not None and _niyet_pipe is not None:
        try:
            res_kat = _kat_pipe(clean_text[:512])[0]
            res_niyet = _niyet_pipe(clean_text[:512])[0]
            category = res_kat["label"].lower()
            cat_conf = float(res_kat["score"])
            intent = res_niyet["label"].lower()
            intent_conf = float(res_niyet["score"])
            sentiment = -0.85 * intent_conf if intent == "sikayet" else (
                0.90 * intent_conf if intent == "tesekkur" else 0.0
            )
            return {
                "category": category,  # type: ignore[typeddict-item]
                "category_confidence": round(cat_conf, 4),
                "intent": intent,  # type: ignore[typeddict-item]
                "intent_confidence": round(intent_conf, 4),
                "sentiment_score": round(sentiment, 4),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                "model_name": _loaded_model_id or "transformer_nlu",
            }
        except Exception:
            pass

    norm = normalize_turkish(clean_text)

    matched_cat: CategoryType = "genel"
    cat_conf = 0.72
    if any(dk in norm for dk in ("dikis", "sokuk", "kirik", "bozuk", "ayipli", "kusurlu", "defolu", "yirtik")):
        matched_cat = "urun"
        cat_conf = 0.92
    else:
        for kw, cat in KEYWORD_CATEGORY_MAP.items():
            if kw in norm:
                matched_cat = cat
                cat_conf = 0.89
                break

    matched_intent: IntentType = "bilgi_talebi"
    intent_conf = 0.75
    sentiment = 0.0

    if any(neg in norm for neg in ("sikayet", "rezalet", "magdur", "berbat", "parami verin", "dolandirici", "hirsiz", "lanet")):
        matched_intent = "sikayet"
        intent_conf = 0.94
        sentiment = -0.90
    elif any(pos in norm for pos in ("tesekkur", "harika", "memnun", "sagol", "mukemmel", "eline saglik", "helal olsun")):
        matched_intent = "tesekkur"
        intent_conf = 0.96
        sentiment = 0.88
    elif any(neg in norm for neg in ("gecik", "gelmedi", "yapilmadi", "kirik", "bozuk", "olmadi", "hata", "sokuk", "ayipli", "cakili", "islanmis", "yanlis", "kesilmis", "yansimadi")):
        matched_intent = "sikayet"
        intent_conf = 0.84
        sentiment = -0.65
    elif any(q in norm for q in ("?", "nasil", "nerede", "kac gun", "ogrenebilir", "sart", "nedir")):
        matched_intent = "bilgi_talebi"
        intent_conf = 0.86
        sentiment = 0.0

    return {
        "category": matched_cat,
        "category_confidence": round(cat_conf, 4),
        "intent": matched_intent,
        "intent_confidence": round(intent_conf, 4),
        "sentiment_score": round(sentiment, 4),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
        "model_name": "rule_based_turkish_nlu",
    }
