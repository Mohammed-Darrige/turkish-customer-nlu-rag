"""
Policy-Grounded Turkish & Multilingual Reply Generator (Ollama + Dual Guardrails).

Generates concise replies grounded in top-3 retrieved chunks, enforces BOTH:
1. Numeric grounding (`ungrounded_numbers`)
2. Citation-span grounding (`ungrounded_citations`) against retrieved `[doc_id:article_no]` tags
3. Explicit localized abstention when retrieval triggers `abstained=True`
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Any

from src.contracts import GenerationResult, LangType, NLUResult, RetrievedEvidence

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_CITATION_RE = re.compile(r"\[([a-z0-9_]+:[^\]]+)\]")


def installed_models(timeout_sec: float = 1.5) -> list[str]:
    """Returns model names available in local Ollama, or [] if unreachable."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", []) if "name" in m]
    except Exception:
        return []


def pick_model(requested: str | None, available: list[str]) -> str | None:
    """Selects requested model if installed, else default, else first available."""
    for candidate in (requested, DEFAULT_MODEL):
        if candidate and candidate in available:
            return candidate
    return available[0] if available else None


def ungrounded_numbers(reply: str, *sources: str) -> list[str]:
    """Returns numbers in reply that do not appear in any source string."""
    allowed = {n.replace(",", ".") for src in sources for n in _NUM_RE.findall(src or "")}
    return [n for n in _NUM_RE.findall(reply) if n.replace(",", ".") not in allowed]


def extract_citations(reply: str) -> list[str]:
    """Extracts inline `[doc_id:article_no]` citation tags from a reply."""
    return [m.strip() for m in _CITATION_RE.findall(reply or "")]


def ungrounded_citations(reply: str, evidences: list[RetrievedEvidence]) -> list[str]:
    """Returns citation tags in `reply` that are not present in `evidences`."""
    allowed = {ev["citation_tag"] for ev in evidences}
    found = extract_citations(reply)
    if not found:
        return ["MISSING_CITATION"]
    return [tag for tag in found if tag not in allowed]


def abstention_reply(lang: LangType) -> str:
    """Returns a clear, honest out-of-scope abstention message."""
    if lang == "ar":
        return (
            "عذراً، هذا السؤال خارج نطاق قانون حماية المستهلك التركي أو سياسات دعم المتجر المتوفرة لدي، "
            "لذلك لا أستطيع تقديم إجابة موثقة بمصدر."
        )
    if lang == "en":
        return (
            "Sorry, this question falls outside the indexed Turkish consumer-law and store support corpus, "
            "so I cannot provide a source-grounded answer."
        )
    return (
        "Bu soru indekslenen Türk tüketici mevzuatı (6502 ve ilgili yönetmelikler) ile mağaza destek rehberinin "
        "kapsamı dışında kaldığı için kaynaklı bir yanıt üretemiyorum."
    )


def build_reply_prompt(
    message: str,
    evidences: list[RetrievedEvidence],
    lang: LangType,
) -> tuple[str, str]:
    """Constructs system and user prompts for multi-chunk cited reply generation."""
    allowed_tags = ", ".join(f"[{ev['citation_tag']}]" for ev in evidences)
    lang_label = {"tr": "Türkçe", "ar": "Arapça", "en": "İngilizce"}[lang]
    system = (
        "Sen bir e-ticaret ve tüketici mevzuatı destek asistanısın. "
        "Yalnızca aşağıda verilen kaynak metinlerdeki bilgileri kullan. "
        "Kaynaklarda yazmayan hiçbir gün, süre, tutar, madde numarası veya kural uydurma. "
        f"Yanıtını {lang_label} dilinde, en fazla 3 kısa ve net cümle olarak yaz. "
        f"Her yanıtın sonunda veya ilgili cümlede yalnızca şu izin verilen kaynak etiketlerinden birini kullan: {allowed_tags}."
    )
    context_blocks = "\n\n".join(
        f"[{ev['citation_tag']}] ({ev['title']}):\n{ev['content']}"
        for ev in evidences
    )
    user = f"Müşteri sorusu: {message}\n\nKaynaklar:\n{context_blocks}"
    return system, user


def _call_ollama_chat(
    system: str,
    user: str,
    model: str,
    timeout_sec: int = 45,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.15, "num_predict": 260},
    }
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return re.sub(r"\s+", " ", data.get("message", {}).get("content", "")).strip()


def template_fallback_reply(
    evidences: list[RetrievedEvidence],
    nlu: NLUResult,
    abstained: bool,
) -> str:
    """Deterministic extractive reply grounded directly in top retrieved evidence."""
    lang = nlu.get("language", "tr")
    if abstained or not evidences:
        return abstention_reply(lang)

    top = evidences[0]
    body_lines = [ln.strip() for ln in top["content"].splitlines() if ln.strip()]
    snippet = " ".join(body_lines[1:]) if len(body_lines) > 1 else top["content"]
    if len(snippet) > 420:
        snippet = snippet[:417].rsplit(" ", 1)[0] + "..."
    return f"{snippet} [{top['citation_tag']}]"


def generate_reply(
    message: str,
    evidences: list[RetrievedEvidence],
    nlu: NLUResult,
    grounding_verified: bool,
    abstained: bool = False,
    use_llm: bool = True,
    requested_model: str | None = None,
) -> GenerationResult:
    """Generates a cited, numerically grounded reply or abstains safely."""
    t0 = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    lang = nlu.get("language", "tr")

    if abstained or not grounding_verified or not evidences:
        reply = abstention_reply(lang)
        return {
            "reply": reply,
            "generation_mode": "abstention_guard",
            "numeric_grounding_passed": True,
            "citation_grounding_passed": True,
            "ungrounded_citations": [],
            "attempts": attempts,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    top_evidences = evidences[:3]
    source_texts = [ev["content"] for ev in top_evidences]

    if use_llm:
        available = installed_models()
        model = pick_model(requested_model, available)
        if model:
            system, base_user = build_reply_prompt(message, top_evidences, lang)
            prompts = [
                base_user,
                base_user + "\n\nÖNEMLİ: Kaynaklarda geçmeyen sayı yazma ve mutlaka [doc_id:Madde N] etiketi ekle.",
            ]
            for prompt in prompts:
                t_call = time.perf_counter()
                try:
                    text = _call_ollama_chat(system, prompt, model)
                    bad_nums = ungrounded_numbers(text, *source_texts, message)
                    bad_cites = ungrounded_citations(text, top_evidences)
                    attempts.append({
                        "ms": round((time.perf_counter() - t_call) * 1000, 1),
                        "ungrounded_numbers": bad_nums,
                        "ungrounded_citations": bad_cites,
                    })
                    if text and not bad_nums and not bad_cites:
                        return {
                            "reply": text,
                            "generation_mode": f"ollama:{model}",
                            "numeric_grounding_passed": True,
                            "citation_grounding_passed": True,
                            "ungrounded_citations": [],
                            "attempts": attempts,
                            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                        }
                except Exception as exc:
                    attempts.append({
                        "ms": round((time.perf_counter() - t_call) * 1000, 1),
                        "error": str(exc),
                    })
                    break

    fallback = template_fallback_reply(top_evidences, nlu, abstained=False)
    bad_cites_fb = ungrounded_citations(fallback, top_evidences)
    return {
        "reply": fallback,
        "generation_mode": "extractive_citation_fallback",
        "numeric_grounding_passed": True,
        "citation_grounding_passed": not bool(bad_cites_fb),
        "ungrounded_citations": bad_cites_fb,
        "attempts": attempts,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
