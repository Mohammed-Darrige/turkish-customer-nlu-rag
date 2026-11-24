# Benchmark Report (Held-Out Evaluation, N = 120)

- **Environment:** Python `3.13.12` on `Windows AMD64`
- **Store Backend:** `in_memory` (`213` chunks)
- **Dense Encoder:** `sentence_transformers:intfloat/multilingual-e5-small`
- **Reranker:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **Corpus SHA-256 (`data/chunks.jsonl`):** `bc99714949dca4b5b04506e2ea7ce932a8d2126cf0051282f3f35c899f1ed575`
- **Held-Out Eval SHA-256 (`data/eval_test.jsonl`):** `4a654e277298fe6dc8cbe02fd4a98cab928b6a6446791f6d48bcc8701e12f371`

## 1. Retrieval, Abstention & Grounding Summary (`eval_test.jsonl`, N = 120)

> **Note on Grounding:** `Num. Grounding` and `Cite. Grounding` are measured under `use_llm=False` (`extractive_citation_fallback`). Live Ollama LLM retry/fallback rates depend on the local model runtime.

| Mode | Hit@1 (N=95) | Recall@3 | Recall@5 | MRR@5 | Abs. Prec | Abs. Rec | Abs. F1 | False Ans. Rate | Num. Grounding | Cite. Grounding | P50 (ms) | P95 (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lexical` | 64.2% | 75.8% | 83.2% | 0.718 | 66.7% | 8.0% | 14.3% | 92.0% | 100.0% | 100.0% | 96.65 | 233.56 |
| `dense` | 63.2% | 80.0% | 84.2% | 0.714 | 60.7% | 68.0% | 64.2% | 32.0% | 100.0% | 100.0% | 110.64 | 242.04 |
| `hybrid` | 68.4% | 83.2% | 89.5% | 0.760 | 62.5% | 80.0% | 70.2% | 20.0% | 100.0% | 100.0% | 72.94 | 116.67 |
| `hybrid_rerank` | 77.9% | 89.5% | 91.6% | 0.840 | 77.4% | 96.0% | 85.7% | 4.0% | 100.0% | 100.0% | 378.54 | 535.09 |

## 2. Turkish-First Scope vs. Cross-Lingual Ablation (`hybrid_rerank`)

| Slice / Bucket | Queries | Hit@1 | Recall@3 | Recall@5 |
| :--- | ---: | ---: | ---: | ---: |
| `colloquial_tr` | 50 | 80.0% | 96.0% | 98.0% |
| `cross_lingual` | 25 | 64.0% | 72.0% | 76.0% |
| `legal_exception` | 20 | 90.0% | 95.0% | 95.0% |
| **`turkish_primary` (`colloquial_tr` + `legal_exception`)** | **70** | **82.9%** | **95.7%** | **97.1%** |
