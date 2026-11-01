"""Where do the systems fail? Breaks a run down by dataset, by retrieval hit/miss, and by
how often two systems get different questions right.

    python eval/analyze.py --run baselines --questions data/questions/*.jsonl

Writes results/<run>/analysis.md. CPU only, reads predictions.jsonl.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.config import CFG  # noqa: E402
from docagent.data import load_questions, read_jsonl  # noqa: E402


def pct(v):
    return f"{100 * sum(v) / len(v):.1f} ({len(v)})" if v else "–"


def analyze(rows: list[dict], qs: dict) -> str:
    by_sys = defaultdict(dict)
    for r in rows:
        if r["qid"] in qs and not r.get("skipped") and r.get("score") is not None:
            by_sys[r["system"]][r["qid"]] = r
    systems = sorted(by_sys)
    out = []

    # 1. accuracy per dataset
    out += ["## Accuracy per dataset", "", "| System | " + " | ".join(
        sorted({q.source for q in qs.values()})) + " |", "|---|" + "---|" * len({q.source for q in qs.values()})]
    for s in systems:
        cells = []
        for src in sorted({q.source for q in qs.values()}):
            cells.append(pct([r["score"] for qid, r in by_sys[s].items() if qs[qid].source == src]))
        out.append(f"| {s} | " + " | ".join(cells) + " |")

    # 2. retrieval hit vs miss (answerable questions with evidence pages)
    out += ["", "## Is it retrieval or reading?", "",
            "Answerable questions only. *Hit* = at least one gold page in the top-k pages handed over.", "",
            "| System | Hit rate | Accuracy when hit | Accuracy when missed |", "|---|---|---|---|"]
    for s in systems:
        hit, miss = [], []
        for qid, r in by_sys[s].items():
            ev = qs[qid].evidence_keys
            if not ev or "unanswerable" in qs[qid].tags:
                continue
            got = {tuple(p) for p in r["retrieved"][:CFG.top_k]} & ev
            (hit if got else miss).append(r["score"])
        n = len(hit) + len(miss)
        if n and not s.startswith("B2"):
            out.append(f"| {s} | {100 * len(hit) / n:.0f}% | {pct(hit)} | {pct(miss)} |")

    # 3. complementarity
    out += ["", "## Who gets what right", ""]
    for a, b in [(x, y) for i, x in enumerate(systems) for y in systems[i + 1:]]:
        common = set(by_sys[a]) & set(by_sys[b])
        sa = {q for q in common if by_sys[a][q]["score"] == 1}
        sb = {q for q in common if by_sys[b][q]["score"] == 1}
        out.append(f"- {a} vs {b} on {len(common)} questions: both right {len(sa & sb)}, only {a} {len(sa - sb)}, "
                   f"only {b} {len(sb - sa)}, either {len(sa | sb)} ({100 * len(sa | sb) / max(1, len(common)):.1f}%)")

    # 4. unanswerable behaviour
    out += ["", "## Saying \"Not answerable\"", "", "| System | On unanswerable questions | False refusals on answerable |",
            "|---|---|---|"]
    from docagent.scoring import says_unanswerable
    for s in systems:
        un = [r for q, r in by_sys[s].items() if "unanswerable" in qs[q].tags]
        an = [r for q, r in by_sys[s].items() if "unanswerable" not in qs[q].tags]
        ref_un = sum(says_unanswerable(r["prediction"]) for r in un)
        ref_an = sum(says_unanswerable(r["prediction"]) for r in an)
        out.append(f"| {s} | refused {ref_un}/{len(un)} | refused {ref_an}/{len(an)} |")

    # 5. who graded
    out += ["", "## Grading", ""]
    for s in systems:
        n_llm = sum(r.get("scored_by") == "llm" for r in by_sys[s].values())
        out.append(f"- {s}: {n_llm} of {len(by_sys[s])} graded by the LLM judge, the rest by rules")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baselines")
    ap.add_argument("--questions", nargs="+", type=Path, required=True)
    args = ap.parse_args()
    qs = {q.qid: q for f in args.questions for q in load_questions(f)}
    rows = read_jsonl(CFG.results_dir / args.run / "predictions.jsonl")
    text = f"# Analysis: {args.run}\n\n" + analyze(rows, qs)
    (CFG.results_dir / args.run / "analysis.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
