"""Run baselines on question files, score them, and write the results table.

    python eval/run_eval.py --questions data/questions/mmlongbench.jsonl --systems B1 B2
    python eval/run_eval.py --questions data/questions/*.jsonl --systems B1 B2 --judge llm --wandb

Results go to results/<run>/predictions.jsonl (one line per system x question) and
results/<run>/summary.md. Reruns with the same --run skip questions already answered,
so a Kaggle session that times out can just be restarted.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.config import CFG  # noqa: E402
from docagent.data import PageStore, Question, load_questions, read_jsonl  # noqa: E402
from docagent.scoring import citation_scores, llm_judge, recall_at_k, rule_score, summary_table  # noqa: E402


def score_row(row: dict, q: Question, judge: str, vlm) -> dict:
    if row.get("skipped"):
        return row
    s = rule_score(q, row["prediction"])
    row["scored_by"] = "rule"
    if s is None and judge == "llm":
        s = llm_judge(q, row["prediction"], lambda p: vlm.generate(p, max_new_tokens=5).text)
        row["scored_by"] = "llm"
    row["score"] = s
    ev = q.evidence_keys
    row["recall_at_k"] = recall_at_k([tuple(r) for r in row["retrieved"]], ev, CFG.top_k) \
        if row["system"].startswith("B1") else None   # B2 has no retrieval step
    row["cite_precision"], row["cite_recall"] = citation_scores([tuple(c) for c in row["cited"]], ev)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", nargs="+", type=Path, required=True)
    ap.add_argument("--systems", nargs="+", default=["B1", "B2"], choices=["B1", "B2"])
    ap.add_argument("--run", default="week1")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions per file (smoke test)")
    ap.add_argument("--judge", choices=["rule", "llm"], default="rule",
                    help="llm: open answers are graded by the same VLM (rule-only leaves them unscored)")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    random.seed(CFG.seed)
    CFG.ensure_dirs()
    out_dir = CFG.results_dir / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    questions: list[Question] = []
    for f in args.questions:
        qs = load_questions(f)
        questions += qs[: args.limit] if args.limit else qs
    by_id = {q.qid: q for q in questions}
    print(f"{len(questions)} questions from {len(args.questions)} file(s)")

    done = {(r["system"], r["qid"]) for r in read_jsonl(pred_path)} if pred_path.exists() else set()

    from docagent.baselines import run_b1, run_b2
    from docagent.retrieve import BGEEmbedder, TextIndex
    from docagent.vlm import VLM

    print("Loading the VLM (first run downloads ~16 GB, a few minutes)...", flush=True)
    vlm = VLM()
    print("VLM ready.", flush=True)
    embedder = BGEEmbedder() if "B1" in args.systems else None
    stores, indexes = {}, {}

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="docagent", name=args.run, config={
            "vlm": CFG.vlm_model, "4bit": CFG.load_in_4bit, "embed": CFG.embed_model, "top_k": CFG.top_k,
            "direct_pages": CFG.direct_vlm_pages, "max_pixels": CFG.max_pixels_per_page, "systems": args.systems})

    with pred_path.open("a", encoding="utf-8") as out:
        for i, q in enumerate(questions, 1):
            store = stores.setdefault(q.collection, PageStore(CFG.pages_dir, q.collection))
            for sysname in args.systems:
                full = {"B1": "B1_text_rag", "B2": "B2_direct_vlm"}[sysname]
                if (full, q.qid) in done:
                    continue
                if sysname == "B1":
                    if q.collection not in indexes:
                        print(f"Building text index for '{q.collection}' ({len(store)} pages)...")
                        indexes[q.collection] = TextIndex.for_collection(store, embedder)
                    row = run_b1(q, indexes[q.collection], vlm)
                else:
                    row = run_b2(q, store, vlm)
                row = score_row(row, q, args.judge, vlm)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                status = "skipped" if row.get("skipped") else f"score {row.get('score')}, {row.get('latency_s', 0):.0f}s"
                print(f"  [{i}/{len(questions)}] {full} {q.qid}: {status}", flush=True)
                if run and row.get("score") is not None:
                    run.log({f"{full}/score": row["score"], f"{full}/latency_s": row["latency_s"]})

    rows = [r for r in read_jsonl(pred_path) if r["qid"] in by_id]
    table = summary_table(rows, by_id, k=CFG.top_k)
    header = f"# Results: {args.run}\n\nVLM: `{CFG.vlm_model}` ({'4-bit' if CFG.load_in_4bit else '16-bit'}), " \
             f"embeddings: `{CFG.embed_model}`, top-k: {CFG.top_k}, direct-VLM pages: {CFG.direct_vlm_pages}\n\n"
    (out_dir / "summary.md").write_text(header + table + "\n", encoding="utf-8")
    print("\n" + table)
    if run:
        import wandb
        run.log({"summary": wandb.Table(columns=["markdown"], data=[[table]])})
        run.finish()


if __name__ == "__main__":
    main()
