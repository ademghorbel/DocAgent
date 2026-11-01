# Analysis: baselines

## Accuracy per dataset

| System | mmlongbench | vidore_v3 |
|---|---|---|
| B1_text_rag | 37.4 (91) | 68.3 (60) |
| B2_direct_vlm | 25.3 (91) | – |
| B3_colqwen_rag | 42.9 (91) | 71.7 (60) |

## Is it retrieval or reading?

Answerable questions only. *Hit* = at least one gold page in the top-k pages handed over.

| System | Hit rate | Accuracy when hit | Accuracy when missed |
|---|---|---|---|
| B1_text_rag | 72% | 45.7 (94) | 36.1 (36) |
| B3_colqwen_rag | 84% | 54.1 (109) | 52.4 (21) |

## Who gets what right

- B1_text_rag vs B2_direct_vlm on 91 questions: both right 21, only B1_text_rag 13, only B2_direct_vlm 2, either 36 (39.6%)
- B1_text_rag vs B3_colqwen_rag on 151 questions: both right 53, only B1_text_rag 22, only B3_colqwen_rag 29, either 104 (68.9%)
- B2_direct_vlm vs B3_colqwen_rag on 91 questions: both right 15, only B2_direct_vlm 8, only B3_colqwen_rag 24, either 47 (51.6%)

## Saying "Not answerable"

| System | On unanswerable questions | False refusals on answerable |
|---|---|---|
| B1_text_rag | refused 18/19 | refused 24/132 |
| B2_direct_vlm | refused 13/19 | refused 23/72 |
| B3_colqwen_rag | refused 11/19 | refused 9/132 |

## Grading

- B1_text_rag: 53 of 151 graded by the LLM judge, the rest by rules
- B2_direct_vlm: 0 of 91 graded by the LLM judge, the rest by rules
- B3_colqwen_rag: 56 of 151 graded by the LLM judge, the rest by rules
