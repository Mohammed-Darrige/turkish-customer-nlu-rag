"""
Official Corpus Fetcher for Turkish Consumer Legislation (mevzuat.gov.tr).

Downloads consolidated statutory texts (PDF/HTML iframe) into `data/raw/*.txt`
and records provenance metadata. Default ingestion reads from `data/raw/` for
deterministic offline builds; `--fetch` refreshes from official endpoints.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"

USER_AGENT = "Mozilla/5.0 (compatible; TurkishCustomerRAG/2.0)"


@dataclass(frozen=True)
class LegalSourceSpec:
    doc_id: str
    title: str
    category: str
    effective_date: str
    source_url: str
    fetch_url: str
    kind: str  # "pdf" | "html_iframe"
    raw_filename: str


LEGAL_SOURCES: tuple[LegalSourceSpec, ...] = (
    LegalSourceSpec(
        doc_id="kanun_6502",
        title="6502 Sayılı Tüketicinin Korunması Hakkında Kanun",
        category="genel",
        effective_date="2026-01-01",
        source_url="https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=6502&MevzuatTur=1&MevzuatTertip=5",
        fetch_url="https://www.mevzuat.gov.tr/MevzuatMetin/1.5.6502.pdf",
        kind="pdf",
        raw_filename="kanun_6502.txt",
    ),
    LegalSourceSpec(
        doc_id="mesafeli_sozlesmeler",
        title="Mesafeli Sözleşmeler Yönetmeliği",
        category="iade",
        effective_date="2026-01-01",
        source_url="https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=20237&MevzuatTur=7&MevzuatTertip=5",
        fetch_url="https://www.mevzuat.gov.tr/anasayfa/MevzuatFihristDetayIframe?MevzuatTur=7&MevzuatNo=20237&MevzuatTertip=5",
        kind="html_iframe",
        raw_filename="mesafeli_sozlesmeler.txt",
    ),
    LegalSourceSpec(
        doc_id="garanti_belgesi",
        title="Garanti Belgesi Yönetmeliği",
        category="urun",
        effective_date="2014-06-13",
        source_url="https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=19782&MevzuatTur=7&MevzuatTertip=5",
        fetch_url="https://www.mevzuat.gov.tr/anasayfa/MevzuatFihristDetayIframe?MevzuatTur=7&MevzuatNo=19782&MevzuatTertip=5",
        kind="html_iframe",
        raw_filename="garanti_belgesi.txt",
    ),
    LegalSourceSpec(
        doc_id="satis_sonrasi",
        title="Satış Sonrası Hizmetler Yönetmeliği",
        category="teknik",
        effective_date="2014-06-13",
        source_url="https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=19783&MevzuatTur=7&MevzuatTertip=5",
        fetch_url="https://www.mevzuat.gov.tr/anasayfa/MevzuatFihristDetayIframe?MevzuatTur=7&MevzuatNo=19783&MevzuatTertip=5",
        kind="html_iframe",
        raw_filename="satis_sonrasi.txt",
    ),
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_pdf_text(content: bytes) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(stream=content, filetype="pdf")
    pages = [page.get_text("text") for page in doc]
    return "\n".join(pages)


def _extract_html_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)


def fetch_source(spec: LegalSourceSpec, timeout_sec: int = 30) -> str:
    """Downloads and extracts plain text for a single legal source."""
    import requests

    resp = requests.get(spec.fetch_url, headers={"User-Agent": USER_AGENT}, timeout=timeout_sec)
    resp.raise_for_status()
    if spec.kind == "pdf":
        text = _extract_pdf_text(resp.content)
    else:
        resp.encoding = resp.encoding or "utf-8"
        text = _extract_html_text(resp.text)

    if len(text.strip()) < 5000 or "MADDE" not in text:
        raise ValueError(
            f"Fetched content for {spec.doc_id} failed sanity check "
            f"(len={len(text)}, has_MADDE={'MADDE' in text})"
        )
    return text


def ensure_raw_sources(fetch: bool = False, raw_dir: Path = RAW_DIR) -> dict[str, dict[str, Any]]:
    """
    Ensures all statutory raw text files exist under `raw_dir`.
    When `fetch=True` (or a file is missing), fetches from `mevzuat.gov.tr`.
    Returns per-source provenance metadata keyed by `doc_id`.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, dict[str, Any]] = {}

    for spec in LEGAL_SOURCES:
        out_path = raw_dir / spec.raw_filename
        if fetch or not out_path.exists():
            text = fetch_source(spec)
            out_path.write_text(text, encoding="utf-8", newline="\n")
        else:
            text = out_path.read_text(encoding="utf-8")

        provenance[spec.doc_id] = {
            "doc_id": spec.doc_id,
            "title": spec.title,
            "category": spec.category,
            "effective_date": spec.effective_date,
            "source_url": spec.source_url,
            "raw_file": str(out_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "raw_sha256": sha256_file(out_path),
            "char_count": len(text),
        }

    return provenance
