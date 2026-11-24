"""
Reproducible Held-Out Evaluation Suite (N=120) -> docs/benchmark.md.

Evaluates `data/eval_test.jsonl` across retrieval modes (`lexical`, `dense`,
`hybrid`, and optionally `hybrid_rerank`), measuring:
- Answerable retrieval: Hit@1, Recall@3, Recall@5, MRR@5, P50/P95 Latency (ms)
- Per-bucket retrieval breakdown: colloquial_tr, cross_lingual, legal_exception
- Full-set abstention: Precision, Recall, F1, False Answer Rate
- Generation grounding (extractive fallback): Numeric & Citation Grounding Pass Rate
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.contracts import RetrievalMode
from src.generator import generate_reply
from src.nlu import classify_text
from src.rag_engine import HybridRAGEngine

CHUNKS_PATH = ROOT_DIR / "data" / "chunks.jsonl"
EVAL_TEST_PATH = ROOT_DIR / "data" / "eval_test.jsonl"
BENCHMARK_OUT = ROOT_DIR / "docs" / "benchmark.md"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_mode(engine: HybridRAGEngine, rows: list[dict[str, Any]], mode: RetrievalMode) -> dict[str, Any]:
    answerable = [r for r in rows if not r["should_abstain"]]

    hit1 = 0
    rec3 = 0
    rec5 = 0
    rr_sum = 0.0
    latencies: list[float] = []

    bucket_stats: dict[str, dict[str, int]] = {}

    tp_abs = 0  # expected abstain=True, predicted abstain=True
    fp_abs = 0  # expected abstain=False, predicted abstain=True
    fn_abs = 0  # expected abstain=True, predicted abstain=False
    tn_abs = 0  # expected abstain=False, predicted abstain=False

    num_pass = 0
    cite_pass = 0

    for row in rows:
        t0 = time.perf_counter()
        res = engine.retrieve(row["query"], top_k=5, mode=mode)
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)

        pred_abs = res["abstained"]
        gold_abs = bool(row["should_abstain"])
        if gold_abs and pred_abs:
            tp_abs += 1
        elif not gold_abs and pred_abs:
            fp_abs += 1
        elif gold_abs and not pred_abs:
            fn_abs += 1
        else:
            tn_abs += 1

        if not gold_abs:
            expected = set(row["expected_citations"])
            top_tags = [ev["citation_tag"] for ev in res["top_evidences"]]
            bucket = row["bucket"]
            if bucket not in bucket_stats:
                bucket_stats[bucket] = {"n": 0, "hit1": 0, "rec3": 0, "rec5": 0}
            bucket_stats[bucket]["n"] += 1

            if top_tags and top_tags[0] in expected:
                hit1 += 1
                bucket_stats[bucket]["hit1"] += 1
            if any(t in expected for t in top_tags[:3]):
                rec3 += 1
                bucket_stats[bucket]["rec3"] += 1
            if any(t in expected for t in top_tags[:5]):
                rec5 += 1
                bucket_stats[bucket]["rec5"] += 1

            for rank, tag in enumerate(top_tags[:5], start=1):
                if tag in expected:
                    rr_sum += 1.0 / rank
                    break

        nlu_res = classify_text(row["query"])
        gen_res = generate_reply(
            message=row["query"],
            evidences=res["top_evidences"],
            nlu=nlu_res,
            grounding_verified=res["grounding_verified"],
            abstained=pred_abs,
            use_llm=False,
        )
        if gen_res["numeric_grounding_passed"]:
            num_pass += 1
        if gen_res["citation_grounding_passed"]:
            cite_pass += 1

    n_ans = max(1, len(answerable))
    n_all = max(1, len(rows))
    n_traps = max(1, tp_abs + fn_abs)
    abs_prec = tp_abs / max(1, tp_abs + fp_abs)
    abs_rec = tp_abs / n_traps
    abs_f1 = (2 * abs_prec * abs_rec / (abs_prec + abs_rec)) if (abs_prec + abs_rec) > 0 else 0.0
    false_answer_rate = fn_abs / n_traps

    return {
        "mode": mode,
        "n_total": len(rows),
        "n_answerable": len(answerable),
        "hit1": hit1 / n_ans,
        "recall3": rec3 / n_ans,
        "recall5": rec5 / n_ans,
        "mrr5": rr_sum / n_ans,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "abs_precision": abs_prec,
        "abs_recall": abs_rec,
        "abs_f1": abs_f1,
        "false_answer_rate": false_answer_rate,
        "numeric_grounding_rate": num_pass / n_all,
        "citation_grounding_rate": cite_pass / n_all,
        "buckets": bucket_stats,
    }


def main() -> None:
    engine = HybridRAGEngine()
    rows = load_jsonl(EVAL_TEST_PATH)

    modes: list[RetrievalMode] = ["lexical"]
    if engine.encoder.is_active:
        modes.extend(["dense", "hybrid"])
    if engine.reranker_model is not None:
        modes.append("hybrid_rerank")

    results = [evaluate_mode(engine, rows, m) for m in modes]

    chunks_sha = sha256_file(CHUNKS_PATH)
    eval_sha = sha256_file(EVAL_TEST_PATH)

    lines = [
        "# Benchmark Report (Held-Out Evaluation, N = 120)",
        "",
        f"- **Environment:** Python `{platform.python_version()}` on `{platform.system()} {platform.machine()}`",
        f"- **Store Backend:** `{engine.store.backend_name}` (`{engine.store.count_chunks()}` chunks)",
        f"- **Dense Encoder:** `{engine.encoder.backend}`",
        f"- **Reranker:** `{engine.reranker_name or 'disabled'}`",
        f"- **Corpus SHA-256 (`data/chunks.jsonl`):** `{chunks_sha}`",
        f"- **Held-Out Eval SHA-256 (`data/eval_test.jsonl`):** `{eval_sha}`",
        "",
        "## 1. Retrieval, Abstention & Grounding Summary (`eval_test.jsonl`, N = 120)",
        "",
        "| Mode | Hit@1 (N=95) | Recall@3 | Recall@5 | MRR@5 | Abs. Prec | Abs. Rec | Abs. F1 | False Ans. Rate | Num. Grounding | Cite. Grounding | P50 (ms) | P95 (ms) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in results:
        lines.append(
            f"| `{r['mode']}` | {r['hit1']*100:.1f}% | {r['recall3']*100:.1f}% | {r['recall5']*100:.1f}% | "
            f"{r['mrr5']:.3f} | {r['abs_precision']*100:.1f}% | {r['abs_recall']*100:.1f}% | "
            f"{r['abs_f1']*100:.1f}% | {r['false_answer_rate']*100:.1f}% | "
            f"{r['numeric_grounding_rate']*100:.1f}% | {r['citation_grounding_rate']*100:.1f}% | "
            f"{r['p50_ms']:.2f} | {r['p95_ms']:.2f} |"
        )

    lines.extend([
        "",
        "## 2. Stratified Answerable Bucket Breakdown (Hybrid Mode)",
        "",
        "| Bucket | Queries | Hit@1 | Recall@3 | Recall@5 |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ])
    hybrid_res = next((r for r in results if r["mode"] == "hybrid"), results[-1])
    for b_name, bs in hybrid_res["buckets"].items():
        n = max(1, bs["n"])
        lines.append(
            f"| `{b_name}` | {bs['n']} | {bs['hit1']/n*100:.1f}% | {bs['rec3']/n*100:.1f}% | {bs['rec5']/n*100:.1f}% |"
        )

    BENCHMARK_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {BENCHMARK_OUT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
