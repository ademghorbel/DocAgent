# Results: baselines

VLM: `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit), embeddings: `BAAI/bge-base-en-v1.5`, top-k: 5, direct-VLM pages: 4

| System | all | text | table | figure | multi-page | unanswerable | Page recall@5 | Citation precision | Latency (s) | Peak GPU (GB) |
|---|---|---|---|---|---|---|---|---|---|---|
| B1_text_rag | 49.7 (151) | 52.0 (100) | 45.7 (35) | 36.0 (50) | 50.6 (77) | 94.7 (19) | 0.51 | 0.48 | 7.8 | 9.7 |
| B2_direct_vlm | 25.3 (91) | 21.4 (42) | 0.0 (15) | 5.4 (37) | 11.4 (35) | 68.4 (19) | – | 0.20 | 13.2 | 16.5 |
| B3_colqwen_rag | 54.3 (151) | 56.0 (100) | 54.3 (35) | 44.0 (50) | 50.6 (77) | 57.9 (19) | 0.64 | 0.55 | 16.0 | 16.5 |

Accuracy in %, number of scored questions in brackets. A question can carry several tags, so tag columns don't add up to `all`.
