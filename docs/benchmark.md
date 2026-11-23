# Benchmark Report (Measured)

- **Corpus:** 18 Turkish e-commerce support policies (`data/policies.json`)
- **Dense Backend:** `sentence_transformers:intfloat/multilingual-e5-small`
- **Reranker:** `disabled`
- **NLU Backend:** `rule_based_turkish_nlu`
- **Environment:** Python 3.13.12 on Windows (AMD64)

## 1. Policy-Level Retrieval Ablation (`data/eval_retrieval.jsonl`, N=21)

| Mode | Policy Hit@1 | Policy Hit@3 | MRR@3 | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| `bm25` | 81.0% (17/21) | 95.2% (20/21) | 0.865 | 0.24 |
| `dense` | 71.4% (15/21) | 85.7% (18/21) | 0.786 | 9.58 |
| `hybrid` | 85.7% (18/21) | 90.5% (19/21) | 0.881 | 8.96 |

## 2. NLU Classification (`data/eval_nlu.jsonl`, N=20)

| Task | Macro F1 | Macro Precision | Macro Recall | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Category (5 classes) | 69.8% | 76.4% | 70.3% | 0.016 |
| Intent (3 classes) | 81.4% | 86.1% | 84.8% | — |

Category confusion matrix labels: `['genel', 'iade', 'kargo', 'teknik', 'urun']`
```
[2, 0, 1, 0, 0]
[0, 3, 2, 0, 0]
[0, 0, 4, 0, 0]
[1, 0, 0, 3, 0]
[1, 1, 0, 0, 2]
```

## 3. Misses in `hybrid` Mode

- Expected `POL_KARGO_01`, got `POL_GENEL_04`: Siparişim aktarma merkezinde bekliyor hareket yok
- Expected `POL_IADE_02`, got `POL_IADE_04`: İç çamaşırının paketini açtım iade edebilir miyim
- Expected `POL_IADE_04`, got `POL_KARGO_01`: İade kodu nedir hangi kargo ile göndermeliyim
