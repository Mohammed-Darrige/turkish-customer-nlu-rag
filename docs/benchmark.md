# Benchmark Report (Held-Out Evaluation, N = 120)

- **Environment:** Python `3.13.12` on `Windows AMD64`
- **Store Backend:** `in_memory` (`213` chunks)
- **Dense Encoder:** `sentence_transformers:intfloat/multilingual-e5-small`
- **Reranker:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **Corpus SHA-256 (`data/chunks.jsonl`):** `e5e5cf8e76298492080a45328d4bd0371fcbca3442a0f2bf7ac11af3d3e11844`
- **Held-Out Eval SHA-256 (`data/eval_test.jsonl`):** `4a654e277298fe6dc8cbe02fd4a98cab928b6a6446791f6d48bcc8701e12f371`

## 1. Retrieval, Abstention & Grounding Summary (`eval_test.jsonl`, N = 120)

| Mode | Hit@1 (N=95) | Recall@3 | Recall@5 | MRR@5 | Abs. Prec | Abs. Rec | Abs. F1 | False Ans. Rate | Num. Grounding | Cite. Grounding | P50 (ms) | P95 (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lexical` | 64.2% | 75.8% | 83.2% | 0.717 | 66.7% | 8.0% | 14.3% | 92.0% | 100.0% | 100.0% | 64.49 | 127.28 |
| `dense` | 63.2% | 80.0% | 84.2% | 0.714 | 60.7% | 68.0% | 64.2% | 32.0% | 100.0% | 100.0% | 104.11 | 295.19 |
| `hybrid` | 67.4% | 83.2% | 89.5% | 0.755 | 62.5% | 80.0% | 70.2% | 20.0% | 100.0% | 100.0% | 72.63 | 132.49 |
| `hybrid_rerank` | 77.9% | 89.5% | 91.6% | 0.840 | 77.4% | 96.0% | 85.7% | 4.0% | 100.0% | 100.0% | 296.37 | 388.62 |

## 2. Stratified Answerable Bucket Breakdown (Hybrid Mode)

| Bucket | Queries | Hit@1 | Recall@3 | Recall@5 |
| :--- | ---: | ---: | ---: | ---: |
| `colloquial_tr` | 50 | 76.0% | 94.0% | 98.0% |
| `cross_lingual` | 25 | 48.0% | 60.0% | 64.0% |
| `legal_exception` | 20 | 70.0% | 85.0% | 100.0% |
