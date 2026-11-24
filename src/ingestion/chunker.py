"""
Structure-Aware Chunker for Turkish Consumer Legislation & Support FAQs.

Splits consolidated statutory texts on `KISIM` / `BÖLÜM` / `MADDE` boundaries
and subdivides long articles on `(1)`, `(2)` fıkra boundaries with 1-fıkra
overlap, injecting contextual hierarchy headers into every chunk so legal
exceptions are never severed from their governing rule.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.contracts import CategoryType, DocumentChunk
from src.ingestion.fetcher import LegalSourceSpec

# Matches PDF page footers like "11993", "11994-1", or standalone page numbers
_PAGE_NO_RE = re.compile(r"^\s*\d{4,5}(?:-\d+)?\s*$")
_KISIM_RE = re.compile(r"^\s*([A-ZÇĞİÖŞÜ]+\s+KISIM)\s*$", re.UNICODE)
_BOLUM_RE = re.compile(r"^\s*([A-ZÇĞİÖŞÜ]+\s+BÖLÜM)\s*$", re.UNICODE)
_ARTICLE_RE = re.compile(
    r"^\s*((?:EK\s+|GEÇİCİ\s+)?MADDE\s+\d+(?:/[A-Z])?)\s*[-–—]?\s*(.*)$",
    re.UNICODE,
)
_FIKRA_RE = re.compile(r"(?m)^\s*(\(\d+\)\s+)")

_CATEGORY_RULES: list[tuple[CategoryType, tuple[str, ...]]] = [
    ("iade", ("cayma", "iade", "mesafeli", "sözleşmeden dönme", "bedel iadesi", "ön bilgilendirme")),
    ("kargo", ("teslim", "kargo", "taşıyıcı", "sevkiyat", "teslimat", "nakliye")),
    ("teknik", ("yetkili servis", "bakım", "onarım", "tamir", "yedek parça", "satış sonrası")),
    ("urun", ("ayıplı", "garanti", "tanıtma", "kullanma kılavuzu", "değişim", "eksik")),
]


def infer_category(text: str, default_category: CategoryType) -> CategoryType:
    lowered = text.lower()
    best_cat: CategoryType = default_category
    best_hits = 0
    for cat, keywords in _CATEGORY_RULES:
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits > best_hits:
            best_hits = hits
            best_cat = cat
    return best_cat


def _clean_lines(raw_text: str) -> list[str]:
    lines: list[str] = []
    for line in raw_text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _PAGE_NO_RE.match(s):
            continue
        # Skip repeated mevzuat.gov.tr PDF running header
        if s.startswith("TÜKETİCİNİN KORUNMASI HAKKINDA KANUN") and len(s) < 55:
            continue
        lines.append(s)
    return lines


def _normalize_article_slug(article_no: str) -> str:
    slug = article_no.lower()
    for src, dst in (("ç", "c"), ("ğ", "g"), ("ı", "i"), ("ö", "o"), ("ş", "s"), ("ü", "u")):
        slug = slug.replace(src, dst)
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug


def _split_fikras(body: str, max_words: int = 340) -> list[str]:
    words = body.split()
    if len(words) <= max_words:
        return [body.strip()] if body.strip() else []

    # Split on numbered paragraphs (1), (2), ...
    parts = _FIKRA_RE.split(body)
    fikras: list[str] = []
    if parts[0].strip():
        fikras.append(parts[0].strip())
    for i in range(1, len(parts), 2):
        marker = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        combined = f"{marker} {content}".strip()
        if combined:
            fikras.append(combined)

    if len(fikras) <= 1:
        # Fallback sentence-boundary split if no (1)/(2) markers exist
        sentences = re.split(r"(?<=[.!?])\s+", body.strip())
        chunks: list[str] = []
        buf: list[str] = []
        buf_words = 0
        for sent in sentences:
            sw = len(sent.split())
            if buf and buf_words + sw > max_words:
                chunks.append(" ".join(buf))
                buf = buf[-1:] if len(buf) > 1 else []
                buf_words = sum(len(x.split()) for x in buf)
            buf.append(sent)
            buf_words += sw
        if buf:
            chunks.append(" ".join(buf))
        return chunks

    chunks = []
    cur: list[str] = []
    cur_words = 0
    for fk in fikras:
        fw = len(fk.split())
        if cur and cur_words + fw > max_words:
            chunks.append("\n".join(cur))
            # 1-fıkra overlap to preserve cross-paragraph context
            overlap = cur[-1:]
            cur = overlap + [fk]
            cur_words = sum(len(x.split()) for x in cur)
        else:
            cur.append(fk)
            cur_words += fw
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def chunk_legal_document(
    raw_text: str,
    spec: LegalSourceSpec,
    max_words: int = 340,
) -> list[DocumentChunk]:
    """Parses a consolidated Turkish legal text into structured, cited chunks."""
    lines = _clean_lines(raw_text)
    current_kisim = ""
    current_bolum = ""
    current_heading = ""
    articles: list[dict[str, str]] = []
    active: dict[str, str] | None = None

    for idx, line in enumerate(lines):
        if _KISIM_RE.match(line):
            current_kisim = line
            current_bolum = ""
            continue
        if _BOLUM_RE.match(line):
            current_bolum = line
            continue

        m = _ARTICLE_RE.match(line)
        if m:
            if active is not None:
                articles.append(active)
            art_no_raw = re.sub(r"\s+", " ", m.group(1)).strip()
            # Normalize "MADDE 48" -> "Madde 48", "EK MADDE 1" -> "Ek Madde 1"
            art_no = " ".join(w.capitalize() for w in art_no_raw.split())
            rest = m.group(2).strip()
            # Capture preceding short title line as article heading if applicable
            heading = current_heading
            section_parts = [p for p in (current_kisim, current_bolum) if p]
            active = {
                "article_no": art_no,
                "heading": heading,
                "section": " > ".join(section_parts) if section_parts else "Genel Hükümler",
                "body": rest,
            }
            current_heading = ""
            continue

        if active is None:
            continue

        # Short title-cased/uppercase line right before next MADDE is usually article heading
        if (
            idx + 1 < len(lines)
            and _ARTICLE_RE.match(lines[idx + 1])
            and len(line) <= 100
            and not line.endswith(".")
        ):
            current_heading = line
            continue

        active["body"] = f"{active['body']}\n{line}".strip() if active["body"] else line

    if active is not None:
        articles.append(active)

    chunks: list[DocumentChunk] = []
    for art in articles:
        body = art["body"].strip()
        if len(body) < 25:
            continue

        sub_texts = _split_fikras(body, max_words=max_words)
        art_slug = _normalize_article_slug(art["article_no"])
        heading_part = f" — {art['heading']}" if art["heading"] else ""
        title = f"{spec.title} | {art['article_no']}{heading_part}"
        citation_tag = f"{spec.doc_id}:{art['article_no']}"
        category = infer_category(f"{art['heading']} {body}", spec.category)  # type: ignore[arg-type]

        for sub_idx, sub_body in enumerate(sub_texts):
            header = f"[{spec.title} | {art['section']} | {art['article_no']}{heading_part}]"
            full_content = f"{header}\n{sub_body}"
            chunks.append({
                "chunk_id": f"{spec.doc_id}__{art_slug}__c{sub_idx}",
                "doc_id": spec.doc_id,
                "title": title,
                "article_no": art["article_no"],
                "citation_tag": citation_tag,
                "category": category,
                "lang": "tr",
                "effective_date": spec.effective_date,
                "source_url": spec.source_url,
                "content": full_content,
                "token_count": len(full_content.split()),
            })

    return chunks


def load_store_ops_chunks(faq_path: Path) -> list[DocumentChunk]:
    """Loads multilingual operational e-commerce support chunks (store_ops_faq)."""
    if not faq_path.exists():
        return []
    entries: list[dict[str, Any]] = json.loads(faq_path.read_text(encoding="utf-8"))
    chunks: list[DocumentChunk] = []
    for entry in entries:
        item_id = str(entry["id"])
        title = str(entry["title"])
        category: CategoryType = entry["category"]
        lang = str(entry.get("lang", "tr"))
        text = str(entry["text"]).strip()
        header = f"[Mağaza Operasyon Rehberi | {item_id} | {title}]"
        full_content = f"{header}\n{text}"
        chunks.append({
            "chunk_id": f"store_ops_faq__{item_id.lower()}__c0",
            "doc_id": "store_ops_faq",
            "title": title,
            "article_no": item_id,
            "citation_tag": f"store_ops_faq:{item_id}",
            "category": category,
            "lang": lang,
            "effective_date": "2026-01-01",
            "source_url": "internal://store_ops_faq",
            "content": full_content,
            "token_count": len(full_content.split()),
        })
    return chunks
