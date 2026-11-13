# Analysis: baselines

## Accuracy per dataset

| System | mmlongbench | own | vidore_v3 |
|---|---|---|---|
| B1_text_rag | 37.4 (91) | 84.8 (33) | 68.3 (60) |
| B2_direct_vlm | 25.3 (91) | 24.2 (33) | – |
| B3_colqwen_rag | 42.9 (91) | 81.8 (33) | 71.7 (60) |
| DA_agent | 50.5 (91) | 90.9 (33) | 60.0 (60) |
| DA_agent_v2 | 51.6 (91) | 87.9 (33) | 61.7 (60) |

## Is it retrieval or reading?

Answerable questions only. *Hit* = at least one gold page in the top-k pages handed over.

| System | Hit rate | Accuracy when hit | Accuracy when missed |
|---|---|---|---|
| B1_text_rag | 73% | 55.2 (116) | 37.2 (43) |
| B3_colqwen_rag | 84% | 61.2 (134) | 44.0 (25) |
| DA_agent | 84% | 60.9 (133) | 34.6 (26) |
| DA_agent_v2 | 84% | 59.7 (134) | 40.0 (25) |

## Who gets what right

- B1_text_rag vs B2_direct_vlm on 124 questions: both right 29, only B1_text_rag 33, only B2_direct_vlm 2, either 64 (51.6%)
- B1_text_rag vs B3_colqwen_rag on 184 questions: both right 76, only B1_text_rag 27, only B3_colqwen_rag 33, either 136 (73.9%)
- B1_text_rag vs DA_agent on 184 questions: both right 87, only B1_text_rag 16, only DA_agent 25, either 128 (69.6%)
- B1_text_rag vs DA_agent_v2 on 184 questions: both right 85, only B1_text_rag 18, only DA_agent_v2 28, either 131 (71.2%)
- B2_direct_vlm vs B3_colqwen_rag on 124 questions: both right 22, only B2_direct_vlm 9, only B3_colqwen_rag 44, either 75 (60.5%)
- B2_direct_vlm vs DA_agent on 124 questions: both right 26, only B2_direct_vlm 5, only DA_agent 50, either 81 (65.3%)
- B2_direct_vlm vs DA_agent_v2 on 124 questions: both right 26, only B2_direct_vlm 5, only DA_agent_v2 50, either 81 (65.3%)
- B3_colqwen_rag vs DA_agent on 184 questions: both right 88, only B3_colqwen_rag 21, only DA_agent 24, either 133 (72.3%)
- B3_colqwen_rag vs DA_agent_v2 on 184 questions: both right 86, only B3_colqwen_rag 23, only DA_agent_v2 27, either 136 (73.9%)
- DA_agent vs DA_agent_v2 on 184 questions: both right 105, only DA_agent 7, only DA_agent_v2 8, either 120 (65.2%)

## Saying "Not answerable"

| System | On unanswerable questions | False refusals on answerable |
|---|---|---|
| B1_text_rag | refused 22/23 | refused 25/161 |
| B2_direct_vlm | refused 17/23 | refused 43/101 |
| B3_colqwen_rag | refused 15/23 | refused 12/161 |
| DA_agent | refused 21/23 | refused 30/161 |
| DA_agent_v2 | refused 21/23 | refused 22/161 |

## Grading

- B1_text_rag: 67 of 184 graded by the LLM judge, the rest by rules
- B2_direct_vlm: 6 of 124 graded by the LLM judge, the rest by rules
- B3_colqwen_rag: 68 of 184 graded by the LLM judge, the rest by rules
- DA_agent: 60 of 184 graded by the LLM judge, the rest by rules
- DA_agent_v2: 63 of 184 graded by the LLM judge, the rest by rules
