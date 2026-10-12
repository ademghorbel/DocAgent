# DocAgent: agentic multimodal RAG for technical documents

When a machine raises an alert, DocAgent reads the equipment's manuals (text, tables and diagrams) and returns a sourced diagnosis with links to the exact pages. It runs on-premise with open-weight models and is benchmarked against standard RAG baselines.

**Status: week 1 of 4.** Data pipeline and the first two baselines are in place. Page-image retrieval (ColQwen) comes in week 2, the agent in week 3, full results in week 4.

## Results

_Week-1 numbers go here once the first Kaggle run is done (`results/week1/summary.md`)._

| System | What it is | Status |
|---|---|---|
| B1: Text RAG | Page text → bge embeddings + FAISS → top-k chunks → VLM | ✅ week 1 |
| B2: Direct VLM | First N page images → VLM, no retrieval | ✅ week 1 |
| B3: Multimodal RAG | ColQwen top-k page images → VLM | week 2 |
| DocAgent | Planner → retrieve → tool (VLM read / table parser / calculator) → verifier | week 3 |

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

Runs on a free Kaggle GPU (T4). Turn on Internet in the notebook settings.

```bash
pip install -r requirements.txt
python scripts/prepare_mmlongbench.py --n 100 --max-docs 15
python scripts/prepare_vidore.py --n 60
python eval/run_eval.py --questions data/questions/mmlongbench.jsonl data/questions/vidore_industrial.jsonl --systems B1 B2
```

Own manuals: add links to `data/manuals.csv`, run `python scripts/download_manuals.py`, write questions in `data/questions/own.jsonl` (see `own.example.jsonl`), then `python scripts/check_questions.py data/questions/own.jsonl`.

Useful switches (environment variables): `DOCAGENT_VLM` (e.g. `Qwen/Qwen3-VL-8B-Instruct`), `DOCAGENT_4BIT=0` for 16-bit, `DOCAGENT_TOP_K`, `DOCAGENT_DIRECT_PAGES`, `DOCAGENT_MAX_PIXELS`.

## Repo layout

```text
docagent/      config, data formats, ingestion, text retrieval, VLM loader, prompts, baselines, scoring
scripts/       benchmark preparation, manual download, question checker
eval/          run_eval.py (runs and scores systems), show_errors.py (error analysis)
data/          question files and the manual list (PDFs and rendered pages are downloaded, not committed)
results/       summary tables per run
tests/         CPU-only tests: `pytest -q`
```

## Notes and limits

- Scanned pages have no text layer. B1 skips their content unless ingestion runs with `--ocr`; the vision systems read them as images. ViDoRe ships its own text layer.
- B2 only runs on questions about one document. Corpus-wide questions are skipped rather than given the gold document, which would leak where the answer is.
- When the LLM judge is used, it is the same model that answers. Hand-check about 20 judgments before trusting the open-answer numbers.
- DocAgent suggests; a human decides. Alerts in the demo are simulated.
