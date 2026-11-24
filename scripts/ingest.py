"""
Deterministic Corpus Ingestion CLI.

Reads consolidated legal snapshots from `data/raw/` (or fetches fresh from
mevzuat.gov.tr with `--fetch`), chunks them structurally, merges `store_ops_faq`,
and writes `data/chunks.jsonl` + `data/corpus_manifest.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.chunker import chunk_legal_document, load_store_ops_chunks
from src.ingestion.fetcher import (
    LEGAL_SOURCES,
    RAW_DIR,
    ensure_raw_sources,
    sha256_file,
)

CHUNKS_PATH = ROOT_DIR / "data" / "chunks.jsonl"
MANIFEST_PATH = ROOT_DIR / "data" / "corpus_manifest.json"
STORE_OPS_PATH = RAW_DIR / "store_ops_faq.json"


def build_corpus(fetch: bool = False) -> dict[str, Any]:
    provenance = ensure_raw_sources(fetch=fetch, raw_dir=RAW_DIR)

    all_chunks = []
    per_doc_counts: dict[str, int] = {}

    for spec in LEGAL_SOURCES:
        raw_text = (RAW_DIR / spec.raw_filename).read_text(encoding="utf-8")
        doc_chunks = chunk_legal_document(raw_text, spec)
        all_chunks.extend(doc_chunks)
        per_doc_counts[spec.doc_id] = len(doc_chunks)

    ops_chunks = load_store_ops_chunks(STORE_OPS_PATH)
    all_chunks.extend(ops_chunks)
    per_doc_counts["store_ops_faq"] = len(ops_chunks)

    # Sanity checks
    seen_ids: set[str] = set()
    for ch in all_chunks:
        if not ch["content"].strip():
            raise ValueError(f"Empty chunk encountered: {ch['chunk_id']}")
        if ch["chunk_id"] in seen_ids:
            raise ValueError(f"Duplicate chunk_id encountered: {ch['chunk_id']}")
        seen_ids.add(ch["chunk_id"])

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8", newline="\n") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    total_tokens = sum(ch["token_count"] for ch in all_chunks)
    manifest: dict[str, Any] = {
        "total_chunks": len(all_chunks),
        "total_tokens": total_tokens,
        "chunks_jsonl_sha256": sha256_file(CHUNKS_PATH),
        "per_doc_chunk_counts": per_doc_counts,
        "sources": provenance,
        "store_ops_faq_sha256": sha256_file(STORE_OPS_PATH) if STORE_OPS_PATH.exists() else None,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structured RAG v2 corpus")
    parser.add_argument("--fetch", action="store_true", help="Refresh raw files from mevzuat.gov.tr")
    args = parser.parse_args()
    manifest = build_corpus(fetch=args.fetch)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
