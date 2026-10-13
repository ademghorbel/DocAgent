# Week 1 (Oct 12–18): data + baselines B1 and B2

**Goal:** a public repo where anyone can reproduce first accuracy numbers for text RAG (B1) and direct VLM (B2) on three question sets.
**Budget:** about 11 hours.

## Milestones

| # | Milestone | Day | Time | Done when |
|---|---|---|---|---|
| M1 | Repo public, Kaggle ready | Mon 12 | 1.5 h | `pytest -q` passes on Kaggle, GPU shows a T4 |
| M2 | Benchmarks prepared | Tue 13 | 1.5 h | `mmlongbench.jsonl` (~100 q) and `vidore_industrial.jsonl` (~60 q) exist, page stores rendered |
| M3 | Own manuals ingested | Wed 14 | 1.5 h | 8 PDFs in `manuals.csv`, ingested, scanned pages noted |
| M4 | Smoke test passes | Thu 15 | 2 h | 5 questions per set run through B1 and B2 with sensible `Answer:` / `Pages:` replies |
| M5 | 30 own questions written | Fri 16 | 2 h | `check_questions.py` reports 0 problems, ~6 questions per tag |
| M6 | Full B1 + B2 run | Sat 17 | 1.5 h (mostly waiting) | `results/week1/summary.md` filled, run logged in W&B |
| M7 | Week wrapped up | Sun 18 | 1 h | README results table, 10 failures read, repo pushed, CEA applications sent |

## Day by day

**Mon: M1.** Create the GitHub repo `docagent` and push this code. On Kaggle: verify your phone number (needed for internet), new notebook, Accelerator **GPU T4 x2**, Internet **on**. Open `notebooks/week1_kaggle.ipynb` there and run the setup cells.

**Tue: M2.** Run the two preparation cells. Then click **Save Version → Save & Run All** once, so the rendered pages become notebook output you can attach later with **Add Input** instead of downloading again.

**Wed: M3.** Pick the manuals: 2 drives, 2 pumps, 1 PLC manual, 3 sensor datasheets. Paste their links into `data/manuals.csv` and run `download_manuals.py`. If a manual is mostly scanned, keep it: that's where B1 should fail and the vision systems shouldn't.

**Thu: M4.** Run the smoke test (`--limit 5`). Open `predictions.jsonl` and read the `raw` replies. Things to check:
- Does the model follow the `Answer:` / `Pages:` format? If not, tighten `docagent/prompts.py`.
- Is B2 running out of memory? Lower `DOCAGENT_DIRECT_PAGES` to 2 or `DOCAGENT_MAX_PIXELS` to `600*28*28`.
- Are the rule scores right? Fix any obviously wrong grading now, before the full run.

**Fri: M5.** Write the 30 questions while flipping through the manuals. Short exact answers (a fault code meaning, a temperature, a current) make scoring reliable. Mix in about 6 cross-document questions ("which sensor…") and 6 unanswerable ones (e.g. warranty, price).

**Sat: M6.** Full run on all three files with `--wandb`. B2 is the slow one (4 page images per question); expect roughly 30–60 minutes in total.

**Sun: M7.**
- Run `show_errors.py` and read 10 failures. Write two or three sentences on patterns, e.g. "B1 misses table questions because flattened tables lose their row/column structure".
- Copy the summary table into the README.
- Push, and pin the repo on your GitHub profile.
- Send the two CEA applications, now that the repo shows real work.
- Optional: a short LinkedIn post with the first table.

## Definition of done

- [ ] Repo public with README, data scripts and question files
- [ ] `pytest -q` passes
- [ ] ~190 questions across 3 sets, every one tagged
- [ ] B1 and B2 results table: accuracy per tag, page recall@5 for B1, latency, GPU memory
- [ ] 10 failures read, notes in the README
- [ ] Applications sent

## What to understand this week

- **How 4-bit loading works** (NF4, compute dtype) and what it costs in accuracy. You'll test that in the week 4 ablation.
- **Why B2 is expensive.** Each page image becomes about 1,000 visual tokens at the default resolution, so 8 pages is roughly an 8k-token prompt.
- **What page recall@k measures** for B1. It's the number you'll compare against ColQwen in week 2.

## If something breaks

| Problem | Fix |
|---|---|
| `bitsandbytes` error on Kaggle | Make sure the accelerator is T4, not P100 |
| CUDA out of memory in B2 | `DOCAGENT_DIRECT_PAGES=4`, or `DOCAGENT_MAX_PIXELS=602112` (768×28×28) |
| Garbage or empty answers in float16 on T4 | Try `DOCAGENT_VLM=Qwen/Qwen2.5-VL-3B-Instruct` to check if it's the model or the setup |
| Kaggle session timed out mid-run | Rerun the same command: finished questions are skipped |
| A manual link returns HTML, not a PDF | The site needs a click-through; download it by hand into `data/pdfs/manuals/` and run `python -m docagent.ingest --collection manuals --pdfs data/pdfs/manuals` |
