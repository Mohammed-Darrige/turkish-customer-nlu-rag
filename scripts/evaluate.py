"""
Honest, Un-Hardcoded Evaluation Harness for Turkish Customer Support RAG & NLU.

Measures:
1. Policy-level retrieval ablation (BM25 vs Dense vs Hybrid RRF) on data/eval_retrieval.jsonl (N=21).
2. NLU Category & Intent classification on data/eval_nlu.jsonl (N=20).
3. Real CPU latency (ms) and writes results to docs/benchmark.md.

Usage:
    python scripts/evaluate.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.contracts import RetrievalMode  # noqa: E402
from src.nlu import classify_text, get_nlu_backend  # noqa: E402
from src.rag_engine import get_rag_engine  # noqa: E402

EVAL_RET_PATH = ROOT_DIR / "data" / "eval_retrieval.jsonl"
EVAL_NLU_PATH = ROOT_DIR / "data" / "eval_nlu.jsonl"
BENCHMARK_MD = ROOT_DIR / "docs" / "benchmark.md"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_retrieval(modes: list[RetrievalMode]) -> list[dict]:
    engine = get_rag_engine()
    cases = _load_jsonl(EVAL_RET_PATH)
    rows: list[dict] = []

    for mode in modes:
        hit1 = 0
        hit3 = 0
        rr_list: list[float] = []
        latencies: list[float] = []
        misses: list[tuple[str, str, str]] = []

        for item in cases:
            q = item["text"]
            expected = item["expected_policy_id"]
            t0 = time.perf_counter()
            res = engine.retrieve(q, top_k=3, mode=mode)
            latencies.append((time.perf_counter() - t0) * 1000)

            ids = [ev["policy_id"] for ev in res["top_evidences"]]
            if ids and ids[0] == expected:
                hit1 += 1
            else:
                misses.append((q, expected, ids[0] if ids else "NONE"))

            if expected in ids:
                hit3 += 1
                rr_list.append(1.0 / (ids.index(expected) + 1))
            else:
                rr_list.append(0.0)

        n = len(cases)
        rows.append({
            "mode": mode,
            "n": n,
            "hit1": round(hit1 / n * 100, 1),
            "hit1_count": f"{hit1}/{n}",
            "hit3": round(hit3 / n * 100, 1),
            "hit3_count": f"{hit3}/{n}",
            "mrr3": round(float(np.mean(rr_list)), 3),
            "mean_ms": round(float(np.mean(latencies)), 2),
            "misses": misses,
        })
    return rows


def evaluate_nlu() -> dict:
    cases = _load_jsonl(EVAL_NLU_PATH)
    y_true_cat, y_pred_cat = [], []
    y_true_int, y_pred_int = [], []
    latencies: list[float] = []

    for item in cases:
        t0 = time.perf_counter()
        res = classify_text(item["text"])
        latencies.append((time.perf_counter() - t0) * 1000)
        y_true_cat.append(item["expected_kategori"])
        y_pred_cat.append(res["category"])
        y_true_int.append(item["expected_niyet"])
        y_pred_int.append(res["intent"])

    p_cat, r_cat, f1_cat, _ = precision_recall_fscore_support(
        y_true_cat, y_pred_cat, average="macro", zero_division=0
    )
    p_int, r_int, f1_int, _ = precision_recall_fscore_support(
        y_true_int, y_pred_int, average="macro", zero_division=0
    )
    labels = sorted(set(y_true_cat + y_pred_cat))
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels)

    return {
        "n": len(cases),
        "backend": get_nlu_backend(),
        "cat_f1": round(float(f1_cat) * 100, 1),
        "cat_p": round(float(p_cat) * 100, 1),
        "cat_r": round(float(r_cat) * 100, 1),
        "int_f1": round(float(f1_int) * 100, 1),
        "int_p": round(float(p_int) * 100, 1),
        "int_r": round(float(r_int) * 100, 1),
        "mean_ms": round(float(np.mean(latencies)), 3),
        "labels": labels,
        "cm": cm.tolist(),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    engine = get_rag_engine()
    modes: list[RetrievalMode] = ["bm25", "dense", "hybrid"]
    if engine.reranker_model is not None:
        modes.append("hybrid_rerank")

    ret_rows = evaluate_retrieval(modes)
    nlu_res = evaluate_nlu()

    print(f"Dense backend: {engine.dense.backend}")
    print(f"NLU backend:   {nlu_res['backend']}")
    print("\n--- Retrieval Ablation (Policy-Level, N=21) ---")
    for r in ret_rows:
        print(
            f"{r['mode']:14s} | Hit@1: {r['hit1']:5.1f}% ({r['hit1_count']}) | "
            f"Hit@3: {r['hit3']:5.1f}% ({r['hit3_count']}) | MRR@3: {r['mrr3']:.3f} | "
            f"Mean: {r['mean_ms']:.2f} ms"
        )

    print("\n--- NLU Held-Out Eval (N=20) ---")
    print(
        f"Category Macro F1: {nlu_res['cat_f1']:.1f}% | "
        f"Intent Macro F1: {nlu_res['int_f1']:.1f}% | Mean: {nlu_res['mean_ms']:.3f} ms"
    )

    BENCHMARK_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Report (Measured)",
        "",
        f"- **Corpus:** {len(engine.policies)} Turkish e-commerce support policies (`data/policies.json`)",
        f"- **Dense Backend:** `{engine.dense.backend}`",
        f"- **Reranker:** `{engine.reranker_name or 'disabled'}`",
        f"- **NLU Backend:** `{nlu_res['backend']}`",
        f"- **Environment:** Python {platform.python_version()} on {platform.system()} ({platform.machine()})",
        "",
        "## 1. Policy-Level Retrieval Ablation (`data/eval_retrieval.jsonl`, N=21)",
        "",
        "| Mode | Policy Hit@1 | Policy Hit@3 | MRR@3 | Mean Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in ret_rows:
        lines.append(
            f"| `{r['mode']}` | {r['hit1']:.1f}% ({r['hit1_count']}) | "
            f"{r['hit3']:.1f}% ({r['hit3_count']}) | {r['mrr3']:.3f} | {r['mean_ms']:.2f} |"
        )

    lines.extend([
        "",
        "## 2. NLU Classification (`data/eval_nlu.jsonl`, N=20)",
        "",
        "| Task | Macro F1 | Macro Precision | Macro Recall | Mean Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| Category (5 classes) | {nlu_res['cat_f1']:.1f}% | {nlu_res['cat_p']:.1f}% | {nlu_res['cat_r']:.1f}% | {nlu_res['mean_ms']:.3f} |",
        f"| Intent (3 classes) | {nlu_res['int_f1']:.1f}% | {nlu_res['int_p']:.1f}% | {nlu_res['int_r']:.1f}% | — |",
        "",
        f"Category confusion matrix labels: `{nlu_res['labels']}`",
        "```",
        *[str(row) for row in nlu_res["cm"]],
        "```",
        "",
        "## 3. Misses in `hybrid` Mode",
        "",
    ])
    hybrid_row = next((r for r in ret_rows if r["mode"] == "hybrid"), None)
    if hybrid_row and hybrid_row["misses"]:
        for q, exp, got in hybrid_row["misses"]:
            lines.append(f"- Expected `{exp}`, got `{got}`: {q}")
    else:
        lines.append("- None (21/21 Hit@1).")

    BENCHMARK_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {BENCHMARK_MD}")


if __name__ == "__main__":
    main()
