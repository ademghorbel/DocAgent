# DocAgent: agentic multimodal RAG for technical documents

When a machine raises an alert, DocAgent reads the equipment's manuals (text, tables and diagrams) and returns a sourced diagnosis with links to the exact pages. It runs on-premise with open-weight models and is benchmarked against standard RAG baselines.

**Status: week 3 of 4.** Three baselines are run and analysed (text RAG, direct VLM, ColQwen page-image RAG); the agent is built and has its first full run (ties B3 at 54.3%). Final results and a demo in week 4.

## Results (151 questions)

Same answering model for every system: Qwen2.5-VL-7B in 4-bit, on a free Kaggle T4 x2.

| System | Accuracy | text | table | figure | multi-page | unanswerable | Page recall@5 | Citation precision | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| B1: Text RAG | 49.7 | 52.0 | 45.7 | 36.0 | 50.6 | **94.7** | 0.51 | 0.48 | 7.8 |
| B2: Direct VLM | 25.3* | 21.4 | 0.0 | 5.4 | 11.4 | 68.4 | – | 0.20 | 13.2 |
| B3: ColQwen RAG | **54.3** | **56.0** | **54.3** | **44.0** | 50.6 | 57.9 | **0.64** | **0.55** | 16.0 |
| DocAgent | **54.3** | 51.0 | 45.7 | 42.0 | 40.3 | 89.5 | 0.59 | 0.52 | 36.7 |

Accuracy in %. \*B2 only answers the 91 single-document questions and sees only the first 4 pages. Full table with counts: [`results/baselines/summary.md`](results/baselines/summary.md); breakdown: [`analysis.md`](results/baselines/analysis.md).

**What the numbers say** ([`analysis.md`](results/baselines/analysis.md))

- **Page images beat page text for retrieval.** ColQwen finds a gold page in the top 5 for 84% of answerable questions, text search for 72%. B3 also answers more table and figure questions right (54 vs 46%, 44 vs 36%), where flattened text loses its structure.
- **Reading, not finding, is now the bottleneck.** When B3 does retrieve a gold page it is right only 54% of the time. Likely cause: the model gets four full pages at once and misses the detail.
- **Text and images fail on different questions.** B1 or B3 is right on 69% of questions, each alone on 50 to 54%.
- **B3 almost never says "Not answerable"** (11 of 19 unanswerable questions, against 18 for B1). Seeing plausible-looking pages, it answers anyway. B1 has the opposite problem: 24 false refusals on answerable questions.
- **Counting across pages is the hardest case** ("how many figures in pages 3 to 10"), for every system.

**Typical failures** ([`errors.txt`](results/baselines/errors.txt))

| Question | Gold | B3 said | Why |
|---|---|---|---|
| How many figures are presented in the section of sensors? | 14 | 7 | Counting over more pages than the 4 it is shown |
| How many times do various sizes of the Eiffel Tower appear in the image? | 20 | 4 | Fine visual detail at 1,000 tokens per page |
| Compared to ViQuAE, how many more QA pairs does SnapNtell have? | 71,980 | 74,680 | Gold page retrieved, wrong values or arithmetic |
| How many more claims does the Wiki Table dataset have than the scientific articles dataset? | 197,940 | 70,613 | Gold page retrieved, wrong values or arithmetic |
| If I drop the green module in Figure 6, what is the drop on 100-shot TACREV? | 0.5% | Not answerable | Needs a figure and a table on different pages |
| Which number shall a MacBook Air user in Mexico call for AppleCare? | 1-800-275-2273 | Not answerable | Gold page retrieved, answer missed in a dense table |
| When will the market open? | Not answerable | 4pm eastern time | Answers from a plausible but unrelated page |
| Compared to MMDU, how many more QA pairs does SnapNtell have? | Not answerable | 70,880 | MMDU is not in the paper; invents a number |
| In which system is the throttle valve placed beneath the fuel injector? | Single Point | Multipoint | Confuses two similar diagrams |
| Does Column G of AFTO Form 70 require the full name? | No, last name only | Yes | Misreads a form instruction |

**Grading.** Short answers are graded by rules. The 109 open answers are graded by the same 7B model; a hand check of 20 agreed 16 times, with errors in both directions that roughly cancel out ([`judge_check.md`](results/baselines/judge_check.md)).

## DocAgent (week 3)

Built on these findings ([`docagent/agent.py`](docagent/agent.py), a LangGraph loop, same 7B model):

```text
retrieve ──> read ──> answer ──> verify ──> done
   ^                               │
   └──── follow-up search ─────────┘   (once, when the verifier says something is missing)
```

1. **Retrieve:** fuse the ColQwen and text rankings (reciprocal rank fusion), since the two find different pages.
2. **Read:** look at one page at a time, image plus its text layer, and copy out the facts that help. A page with nothing useful returns `NONE`.
3. **Answer** from those notes, with the one or two most useful page images for a last look.
4. **Verify:** the model checks the answer against the notes. If something is missing, it writes a follow-up search and the agent reads more pages once. If no page had anything useful, the answer is "Not answerable", which targets B3's main weakness.

Every answer keeps a trace (pages read, notes, verifier decision) for the demo and for error analysis.

## Systems

| System | What it is | Status |
|---|---|---|
| B1: Text RAG | Page text → bge embeddings + FAISS → top-k chunks → VLM | ✅ week 1 |
| B2: Direct VLM | First 4 page images → VLM, no retrieval (8 does not fit on a 16 GB T4) | ✅ week 1 |
| B3: Multimodal RAG | ColQwen2 late-interaction page retrieval → top 4 page images → VLM | ✅ week 2 |
| DocAgent | Fused retrieval → page-by-page reading → answer → verifier, with one follow-up search | ✅ week 3, first run done |

The same VLM (Qwen2.5-VL-7B, 4-bit) answers for every system, so differences come from retrieval and the agent, not the model.

## Evaluation data

| Set | Source | Questions | Search scope |
|---|---|---|---|
| MMLongBench-Doc | [yubo2333/MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc), technical doc types only | ~100 | One document per question |
| ViDoRe V3 industrial | [vidore/vidore_v3_industrial](https://huggingface.co/datasets/vidore/vidore_v3_industrial), USAF technical orders, English | ~60 | Whole corpus (27 manuals, ~5,240 pages) |
| Own manuals | Public drive, pump, PLC and sensor manuals listed in `data/manuals.csv` | ~30 | All manuals |

Every question is tagged `text`, `table`, `figure`, `multi-page` or `unanswerable`, so results are reported per evidence type.

**Metrics:** answer accuracy (rules for short answers, LLM judge for open answers), page recall@k, citation precision, latency, peak GPU memory.

## Reproduce

Runs on a free Kaggle GPU (T4 x2). Easiest: import `notebooks/docagent-kaggle.ipynb`, turn on Internet, and click **Save Version → Save & Run All**; it runs everything in the background. By hand:

```bash
pip install -r requirements.txt
python scripts/prepare_mmlongbench.py --n 100 --max-docs 15
python scripts/prepare_vidore.py --n 60
python scripts/index_pages.py --questions data/questions/*.jsonl      # ColQwen page index for B3
python eval/run_eval.py --questions data/questions/*.jsonl --systems B1 B2 B3 DA --judge llm --run baselines
python eval/analyze.py --run baselines --questions data/questions/*.jsonl   # breakdown, CPU only
```

Own manuals: add links to `data/manuals.csv`, run `python scripts/download_manuals.py`, write questions in `data/questions/own.jsonl` (see `own.example.jsonl`), then `python scripts/check_questions.py data/questions/own.jsonl`.

Useful switches (environment variables): `DOCAGENT_VLM` (e.g. `Qwen/Qwen3-VL-8B-Instruct`), `DOCAGENT_4BIT=0` for 16-bit, `DOCAGENT_TOP_K`, `DOCAGENT_DIRECT_PAGES`, `DOCAGENT_MAX_PIXELS`.

## Repo layout

```text
docagent/      config, data formats, ingestion, text and visual retrieval, VLM loader, prompts, baselines,
               the agent (agent.py), scoring
scripts/       benchmark preparation, manual download, question checker
eval/          run_eval.py (runs and scores systems), analyze.py and show_errors.py (error analysis),
               rescore.py (re-parse and rescore a run without a GPU)
data/          question files and the manual list (PDFs and rendered pages are downloaded, not committed)
results/       summary, analysis, failures and judge check per run
tests/         CPU-only tests: `pytest -q`
```

## Notes and limits

- Scanned pages have no text layer. B1 skips their content unless ingestion runs with `--ocr`; the vision systems read them as images. ViDoRe ships its own text layer.
- B2 only runs on questions about one document. Corpus-wide questions are skipped rather than given the gold document, which would leak where the answer is.
- When the LLM judge is used, it is the same model that answers. 20 judgments were checked by hand (80% agreement, see `results/baselines/judge_check.md`).
- B2 sees only the first 4 pages of a document (8 does not fit in memory), so it is a lower bound for long-context VLMs, not a verdict on them.
- MMLongBench has 91 questions, not 100: one PDF (`mi_phone.pdf`) failed the download check and its 9 questions were dropped.
- DocAgent suggests; a human decides. Alerts in the demo are simulated.
