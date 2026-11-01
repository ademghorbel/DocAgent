"""Re-read the raw model replies of a run with the current parser and rescore it, without a GPU.

Use after a fix to parse_reply or the rule scorer. Rows graded by the LLM judge keep their grade
(re-judging needs the model), unless the rules can now decide them. Rewrites predictions.jsonl
(the old file is kept as predictions.before_rescore.jsonl) and summary.md.

    python eval/rescore.py --run baselines --questions data/questions/*.jsonl
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.config import CFG  # noqa: E402
from docagent.data import load_questions, read_jsonl, write_jsonl  # noqa: E402
from docagent.prompts import parse_reply  # noqa: E402
from docagent.scoring import citation_scores, rule_score, summary_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baselines")
    ap.add_argument("--questions", nargs="+", type=Path, required=True)
    args = ap.parse_args()

    qs = {q.qid: q for f in args.questions for q in load_questions(f)}
    folder = CFG.results_dir / args.run
    path = folder / "predictions.jsonl"
    rows = read_jsonl(path)
    changed = flipped = 0
    for r in rows:
        if r.get("skipped") or r["qid"] not in qs or not r.get("raw"):
            continue
        q = qs[r["qid"]]
        answer, cited = parse_reply(r["raw"])
        if answer == r["prediction"] and [list(c) for c in cited] == r["cited"]:
            continue
        changed += 1
        r["prediction"], r["cited"] = answer, [list(c) for c in cited]
        r["cite_precision"], r["cite_recall"] = citation_scores(cited, q.evidence_keys)
        s = rule_score(q, answer)
        if s is not None:
            flipped += s != r.get("score")
            r["score"], r["scored_by"] = s, "rule"

    backup = folder / "predictions.before_rescore.jsonl"
    if not backup.exists():
        shutil.copy(path, backup)
    write_jsonl(path, rows)
    table = summary_table([r for r in rows if r["qid"] in qs], qs, k=CFG.top_k)
    summary = folder / "summary.md"
    head = summary.read_text(encoding="utf-8").split("| System |")[0] if summary.exists() else f"# Results: {args.run}\n\n"
    summary.write_text(head + table + "\n", encoding="utf-8")
    print(f"{changed} replies parsed differently, {flipped} scores changed\n\n{table}")


if __name__ == "__main__":
    main()
