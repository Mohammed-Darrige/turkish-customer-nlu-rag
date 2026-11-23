"""
Policy-Grounded Turkish Reply Generator (Local Ollama + Numeric Verification).

Generates concise Turkish customer support replies grounded in the retrieved
policy text. Enforces strict numeric grounding (rejects fabricated days,
amounts, or codes not present in the policy or user message) and falls back
to a deterministic policy-grounded template when Ollama is unavailable.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Any

from src.contracts import GenerationResult, NLUResult, RetrievedEvidence

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


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


def build_reply_prompt(
    message: str,
    policy_title: str,
    policy_text: str,
) -> tuple[str, str]:
    """Constructs system and user prompts for strict policy-grounded Turkish reply."""
    system = (
        "Sen bir e-ticaret mağazasının müşteri destek asistanısın. "
        "Yalnızca verilen mağaza politikasındaki bilgileri kullan. "
        "Politikada yazmayan hiçbir gün, süre, tutar, kod veya kural uydurma. "
        "En fazla 3 kısa, net ve saygılı Türkçe cümle yaz; müşteriye 'siz' diye hitap et. "
        "Başlık, madde işareti veya emoji kullanma."
    )
    user = (
        f"Müşteri mesajı: {message}\n\n"
        f"İlgili politika ({policy_title}):\n{policy_text}"
    )
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
        "options": {"temperature": 0.2, "num_predict": 240},
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
    evidence: RetrievedEvidence | None,
    nlu: NLUResult,
    grounding_verified: bool,
) -> str:
    """Deterministic Turkish reply grounded directly in the retrieved policy."""
    if nlu["intent"] == "tesekkur" and (not evidence or evidence["policy_id"] == "POL_GENEL_02"):
        return (
            "Geri bildiriminiz için çok teşekkür ederiz. "
            "Memnun kalmanıza sevindik, her zaman yardımcı olmaktan mutluluk duyarız."
        )
    if not grounding_verified or evidence is None:
        return (
            "Mesajınızı aldık ancak bu konuyla ilgili net bir mağaza politikası eşleşmesi bulamadık. "
            "Sipariş, kargo, iade veya ürün konularındaki sorunuzu biraz daha detaylandırabilir misiniz?"
        )
    return f"Merhaba, ilgili kuralımız ({evidence['title']}) kapsamında: {evidence['content']}"


def generate_reply(
    message: str,
    evidence: RetrievedEvidence | None,
    nlu: NLUResult,
    grounding_verified: bool,
    use_llm: bool = True,
    requested_model: str | None = None,
) -> GenerationResult:
    """Generates a policy-grounded reply via Ollama or deterministic template fallback."""
    t0 = time.perf_counter()
    attempts: list[dict[str, Any]] = []

    policy_title = evidence["title"] if evidence else ""
    policy_text = evidence["content"] if evidence else ""

    if use_llm and grounding_verified and evidence is not None:
        available = installed_models()
        model = pick_model(requested_model, available)
        if model:
            system, base_user = build_reply_prompt(message, policy_title, policy_text)
            prompts = [
                base_user,
                base_user + "\n\nÖNEMLİ: Politikada geçmeyen hiçbir sayı yazma.",
            ]
            for prompt in prompts:
                t_call = time.perf_counter()
                try:
                    text = _call_ollama_chat(system, prompt, model)
                    bad = ungrounded_numbers(text, policy_text, message)
                    attempts.append({
                        "ms": round((time.perf_counter() - t_call) * 1000, 1),
                        "ungrounded_numbers": bad,
                    })
                    if text and not bad:
                        return {
                            "reply": text,
                            "generation_mode": f"ollama:{model}",
                            "numeric_grounding_passed": True,
                            "attempts": attempts,
                            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                        }
                except Exception as exc:
                    attempts.append({
                        "ms": round((time.perf_counter() - t_call) * 1000, 1),
                        "error": str(exc),
                    })
                    break

    fallback = template_fallback_reply(evidence, nlu, grounding_verified)
    return {
        "reply": fallback,
        "generation_mode": "policy_template_fallback",
        "numeric_grounding_passed": True,
        "attempts": attempts,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
