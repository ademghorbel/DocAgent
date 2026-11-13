# Results: baselines

VLM: `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit), embeddings: `BAAI/bge-base-en-v1.5`, top-k: 5, direct-VLM pages: 4

| System | all | text | table | figure | multi-page | unanswerable | Page recall@5 | Citation precision | Latency (s) | Peak GPU (GB) |
|---|---|---|---|---|---|---|---|---|---|---|
| B1_text_rag | 56.0 (184) | 56.9 (116) | 56.5 (46) | 35.8 (53) | 52.5 (80) | 95.7 (23) | 0.55 | 0.48 | 8.0 | 9.7 |
| B2_direct_vlm | 25.0 (124) | 22.4 (58) | 0.0 (26) | 5.0 (40) | 13.2 (38) | 73.9 (23) | – | 0.14 | 13.6 | 16.6 |
| B3_colqwen_rag | 59.2 (184) | 59.5 (116) | 58.7 (46) | 47.2 (53) | 51.2 (80) | 65.2 (23) | 0.68 | 0.52 | 15.9 | 16.6 |
| DA_agent | 60.9 (184) | 57.8 (116) | 52.2 (46) | 45.3 (53) | 42.5 (80) | 91.3 (23) | 0.65 | 0.51 | 37.2 | 9.5 |
| DA_agent_v2 | 61.4 (184) | 58.6 (116) | 45.7 (46) | 45.3 (53) | 43.8 (80) | 91.3 (23) | 0.66 | 0.53 | 50.0 | 17.0 |

Accuracy in %, number of scored questions in brackets. A question can carry several tags, so tag columns don't add up to `all`.
