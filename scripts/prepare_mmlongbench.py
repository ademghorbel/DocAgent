"""Sample MMLongBench-Doc questions, download only the PDFs they need, and ingest them.

    python scripts/prepare_mmlongbench.py --n 100 --max-docs 15

MMLongBench-Doc has 1,091 questions over 135 PDFs. Every question is about one document,
so the collection is searched per document (doc_ids is set).
"""
from __future__ import annotations

import argparse
import ast
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.config import CFG  # noqa: E402
from docagent.data import UNANSWERABLE, PageStore, Question, save_questions  # noqa: E402
from docagent.ingest import ingest_pdf  # noqa: E402

REPO = "yubo2333/MMLongBench-Doc"
COLLECTION = "mmlongbench"
# Document types that look most like technical documentation.
TECHNICAL_TYPES = ("Tutorial/Workshop", "Guidebook", "Academic paper", "Administration/Industry file")


def _as_list(v):
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            v = [v]
    return list(v or [])


def tags_for(sources: list[str], pages: list[int], answer: str) -> list[str]:
    tags = set()
    for s in sources:
        s_low = s.lower()
        if "table" in s_low:
            tags.add("table")
        elif any(k in s_low for k in ("chart", "figure", "image")):
            tags.add("figure")
        elif "text" in s_low:
            tags.add("text")
    if len(set(pages)) > 1:
        tags.add("multi-page")
    if answer.strip().lower() == UNANSWERABLE.lower():
        tags = {"unanswerable"}
    return sorted(tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of questions to keep")
    ap.add_argument("--max-docs", type=int, default=15, help="cap on PDFs to download and render")
    ap.add_argument("--all-types", action="store_true", help="don't restrict to technical-looking doc types")
    ap.add_argument("--out", type=Path, default=CFG.data_dir / "questions" / "mmlongbench.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    ds = load_dataset(REPO, split="train")
    rows = [r for r in ds if args.all_types or any(r["doc_type"].startswith(t) for t in TECHNICAL_TYPES)]
    print(f"{len(rows)} candidate questions after doc-type filter")

    rng = random.Random(CFG.seed)
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)
    docs = sorted(by_doc)
    rng.shuffle(docs)

    picked_docs, picked = [], []
    for d in docs:
        if len(picked) >= args.n or len(picked_docs) >= args.max_docs:
            break
        picked_docs.append(d)
        picked.extend(by_doc[d])
    rng.shuffle(picked)
    picked = picked[: args.n]
    picked_docs = sorted({r["doc_id"] for r in picked})

    # Download first. A few PDFs in the dataset repo fail Hugging Face's size check; retry once,
    # then drop that document and its questions rather than crash the whole preparation.
    pdf_dir = CFG.pdf_dir / COLLECTION
    local_pdfs: dict[str, Path] = {}
    for d in picked_docs:
        for force in (False, True):
            try:
                local_pdfs[d] = Path(hf_hub_download(REPO, f"documents/{d}", repo_type="dataset",
                                                     local_dir=pdf_dir, force_download=force))
                break
            except OSError as e:
                if force:
                    print(f"  SKIP {d}: download failed twice ({str(e).splitlines()[0]})")
    dropped = [r for r in picked if r["doc_id"] not in local_pdfs]
    picked = [r for r in picked if r["doc_id"] in local_pdfs]
    picked_docs = sorted(local_pdfs)
    if dropped:
        print(f"  Dropped {len(dropped)} questions whose PDF could not be downloaded")

    questions = []
    bad_pages = 0
    for i, r in enumerate(picked):
        pages = [int(p) for p in _as_list(r["evidence_pages"])]
        # Pages are 1-based in MMLongBench ("the chart on page 14" -> [14]), but a few
        # annotations say page 0. Drop those: they would count as misses for every system.
        bad_pages += sum(p < 1 for p in pages)
        pages = [p for p in pages if p >= 1]
        sources = [str(s) for s in _as_list(r["evidence_sources"])]
        questions.append(Question(
            qid=f"mmlb-{i:04d}",
            collection=COLLECTION,
            question=r["question"],
            answer=str(r["answer"]),
            answer_format=r["answer_format"],
            doc_ids=[r["doc_id"]],
            evidence=[{"doc_id": r["doc_id"], "page": p} for p in pages],
            tags=tags_for(sources, pages, str(r["answer"])),
            source="mmlongbench",
            extra={"doc_type": r["doc_type"], "evidence_sources": sources},
        ))
    save_questions(args.out, questions)
    if bad_pages:
        print(f"  Removed {bad_pages} evidence page(s) numbered 0 (annotation error in the benchmark)")
    print(f"Saved {len(questions)} questions over {len(picked_docs)} documents -> {args.out}")
    print("Tag counts:", dict(Counter(t for q in questions for t in q.tags)))

    store = PageStore(CFG.pages_dir, COLLECTION)
    store.root.mkdir(parents=True, exist_ok=True)
    for d in picked_docs:
        s = ingest_pdf(local_pdfs[d], store, doc_id=d)
        store.save()  # save after each document so a crash never loses finished work
        print(f"  {d}: {s['pages']} pages")
    print(f"Page store: {len(store)} pages in {store.root}")


if __name__ == "__main__":
    main()
