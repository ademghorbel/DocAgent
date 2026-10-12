"""Check a question file against its page store before running anything expensive.

    python scripts/check_questions.py data/questions/own.jsonl

Catches the usual mistakes: wrong doc_id, page numbers off by one, missing tags.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.config import CFG  # noqa: E402
from docagent.data import TAGS, UNANSWERABLE, PageStore, load_questions  # noqa: E402


def check(path: Path) -> int:
    questions = load_questions(path)
    stores: dict[str, PageStore] = {}
    problems = 0
    seen = set()
    for q in questions:
        store = stores.setdefault(q.collection, PageStore(CFG.pages_dir, q.collection))
        issues = []
        if q.qid in seen:
            issues.append("duplicate qid")
        seen.add(q.qid)
        if not len(store):
            issues.append(f"collection '{q.collection}' has no ingested pages")
        for d in q.doc_ids:
            if len(store) and d not in store.doc_ids():
                issues.append(f"unknown doc_id {d}")
        for e in q.evidence:
            if len(store) and not store.get(e["doc_id"], e["page"]):
                issues.append(f"evidence page not found: {e['doc_id']} p{e['page']}")
        bad_tags = [t for t in q.tags if t not in TAGS]
        if bad_tags or not q.tags:
            issues.append(f"tags should be from {TAGS}, got {q.tags}")
        if q.answer == UNANSWERABLE and q.evidence:
            issues.append("unanswerable question should have no evidence pages")
        if q.answer != UNANSWERABLE and not q.evidence:
            issues.append("answerable question needs at least one evidence page")
        if issues:
            problems += 1
            print(f"{q.qid}: " + "; ".join(issues))
    print(f"\n{len(questions)} questions, {problems} with problems")
    print("Tags:", dict(Counter(t for q in questions for t in q.tags)))
    return problems


if __name__ == "__main__":
    sys.exit(1 if check(Path(sys.argv[1])) else 0)
