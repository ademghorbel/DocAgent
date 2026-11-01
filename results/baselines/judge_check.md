# Hand check of the LLM judge (baselines run)

109 open answers (ViDoRe) were graded by the LLM judge, which is the same Qwen2.5-VL-7B that answers.
20 of them were checked by hand: 10 the judge marked correct, 10 it marked wrong (random, seed 7).

**Agreement: 16 of 20 (80%).**

| # | System | Question | Judge | Hand | Note |
|---|---|---|---|---|---|
| 1 | B1 | vidore-254 | ✔ | ✔ | 175 PSI, same value |
| 2 | B3 | vidore-217 | ✔ | ✔ | "Yes" matches; adds trailers, which the gold says the manual doesn't mention (borderline) |
| 3 | B1 | vidore-37 | ✔ | ✔ | 35 feet |
| 4 | B3 | vidore-89 | ✔ | ✘ | Fuel manual side is wrong (forms, not spotters and hand signals) |
| 5 | B3 | vidore-5 | ✔ | ✔ | Ball valve position and cleaning covered; grounding missing (borderline) |
| 6 | B1 | vidore-222 | ✔ | ✔ | Manufacturing week and year |
| 7 | B1 | vidore-221 | ✔ | ✔ | |
| 8 | B3 | vidore-153 | ✔ | ✔ | Visual inspection |
| 9 | B1 | vidore-85 | ✔ | ✔ | TO 00-25-234 |
| 10 | B1 | vidore-156 | ✔ | ✔ | |
| 11 | B3 | vidore-120 | ✘ | ✔ | "Iron Percent": iron is right, the judge punished the terse answer |
| 12 | B3 | vidore-106 | ✘ | ✘ | Right tape, misses temperature range and fluid resistance |
| 13 | B1 | vidore-5 | ✘ | ✘ | Wrong considerations |
| 14 | B1 | vidore-188 | ✘ | ✘ | Misses the static grounding step |
| 15 | B1 | vidore-82 | ✘ | ✘ | Names no framework (AFM 91-201, NFPA 1500, ...) |
| 16 | B3 | vidore-42 | ✘ | ✔ | Injury, death, health hazards: the gold's key consequences |
| 17 | B1 | vidore-32 | ✘ | ✘ | |
| 18 | B1 | vidore-181 | ✘ | ✔ | Says Type K, same as the gold |
| 19 | B3 | vidore-247 | ✘ | ✘ | Says full name; gold says last name only |
| 20 | B3 | vidore-252 | ✘ | ✘ | Ambulance paint scheme missing (borderline) |

## What it means

- The judge errs both ways: 1 false "correct" in 10, 3 false "wrong" in 10. It is too strict with short
  answers and with answers that add detail.
- Projected over the 109 judged answers (84 correct, 25 wrong): about 84 × 0.9 + 25 × 0.3 ≈ 83 truly
  correct, so the open-answer accuracy is close to unbiased, within the noise of a 20-item check.
- Not fixed for this run: the judge sees the same prompt for every system, so the comparison between
  systems stays fair. A stronger, separate judge model is planned for the final results.
- Known gold issue: vidore-133's reference looks inverted ("not uniformly distributed" marked acceptable).
