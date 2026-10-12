"""Print wrong answers side by side with the gold answer, for error analysis.

    python eval/show_errors.py --run week1 --questions data/questions/*.jsonl --system B1_text_rag -n 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.config import CFG  # noqa: E402
from docagent.data import load_questions, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="week1")
    ap.add_argument("--questions", nargs="+", type=Path, required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("-n", type=int, default=10)
    args = ap.parse_args()

    qs = {q.qid: q for f in args.questions for q in load_questions(f)}
    rows = [r for r in read_jsonl(CFG.results_dir / args.run / "predictions.jsonl")
            if r.get("score") == 0 and r["qid"] in qs and (not args.system or r["system"] == args.system)]
    for r in rows[: args.n]:
        q = qs[r["qid"]]
        print(f"--- {r['system']} | {q.qid} | tags {q.tags}")
        print(f"Q:     {q.question}")
        print(f"Gold:  {q.answer}   (pages {sorted(q.evidence_keys)})")
        print(f"Pred:  {r['prediction']}")
        print(f"Cited: {r['cited']}   Retrieved: {r['retrieved'][:5]}")
    print(f"\n{len(rows)} wrong answers in total")


if __name__ == "__main__":
    main()
