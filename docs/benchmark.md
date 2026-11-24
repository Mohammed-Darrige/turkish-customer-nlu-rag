# Benchmark Report (Held-Out Evaluation, N = 120)

- **Environment:** Python `3.13.12` on `Windows AMD64`
- **Store Backend:** `in_memory` (`213` chunks)
- **Dense Encoder:** `sentence_transformers:intfloat/multilingual-e5-small`
- **Reranker:** `disabled`
- **Corpus SHA-256 (`data/chunks.jsonl`):** `d14ae28eb6a9f9a6bb9d77497a3a289233e33886c82477a0f10ceb559b62d896`
- **Held-Out Eval SHA-256 (`data/eval_test.jsonl`):** `4a654e277298fe6dc8cbe02fd4a98cab928b6a6446791f6d48bcc8701e12f371`

## 1. Retrieval, Abstention & Grounding Summary (`eval_test.jsonl`, N = 120)

| Mode | Hit@1 (N=95) | Recall@3 | Recall@5 | MRR@5 | Abs. Prec | Abs. Rec | Abs. F1 | False Ans. Rate | Num. Grounding | Cite. Grounding | P50 (ms) | P95 (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lexical` | 53.7% | 65.3% | 72.6% | 0.606 | 66.7% | 8.0% | 14.3% | 92.0% | 100.0% | 100.0% | 3.15 | 4.23 |
| `dense` | 53.7% | 67.4% | 75.8% | 0.617 | 59.3% | 64.0% | 61.5% | 36.0% | 100.0% | 100.0% | 12.06 | 20.46 |
| `hybrid` | 58.9% | 72.6% | 80.0% | 0.665 | 55.9% | 76.0% | 64.4% | 24.0% | 100.0% | 100.0% | 13.79 | 19.66 |

## 2. Stratified Answerable Bucket Breakdown (Hybrid Mode)

| Bucket | Queries | Hit@1 | Recall@3 | Recall@5 |
| :--- | ---: | ---: | ---: | ---: |
| `colloquial_tr` | 50 | 64.0% | 82.0% | 88.0% |
| `cross_lingual` | 25 | 48.0% | 56.0% | 64.0% |
| `legal_exception` | 20 | 60.0% | 70.0% | 80.0% |
