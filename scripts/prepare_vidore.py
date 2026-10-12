"""Sample English queries from ViDoRe V3 (industrial subset) and build the page store.

    python scripts/prepare_vidore.py --n 60
    python scripts/prepare_vidore.py --n 60 --no-images   # text only, enough for B1

The industrial subset is 27 USAF technical orders (~5,240 pages). Queries are about the whole
corpus, not one document, so doc_ids stays empty: retrieval has to find the right manual too.
The corpus ships page images plus a markdown text layer, so no OCR is needed.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.config import CFG  # noqa: E402
from docagent.data import Page, PageStore, Question, safe_name, save_questions  # noqa: E402

COLLECTION = "vidore_industrial"


def tags_for(content_types: list[str], n_pages: int) -> list[str]:
    tags = set()
    for c in content_types or []:
        c = c.lower()
        if "table" in c:
            tags.add("table")
        elif any(k in c for k in ("image", "chart", "infographic", "figure", "diagram")):
            tags.add("figure")
        elif "text" in c:
            tags.add("text")
    if n_pages > 1:
        tags.add("multi-page")
    return sorted(tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="vidore/vidore_v3_industrial")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--no-images", action="store_true", help="skip saving page images (B1 only)")
    ap.add_argument("--out", type=Path, default=CFG.data_dir / "questions" / "vidore_industrial.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset

    queries = load_dataset(args.repo, "queries", split="test")
    qrels = load_dataset(args.repo, "qrels", split="test")
    # Only text columns here; images are read later, row by row, to keep memory low.
    corpus = load_dataset(args.repo, "corpus", split="test")
    meta = corpus.select_columns(["corpus_id", "doc_id", "page_number_in_doc"])
    page_of = {r["corpus_id"]: (r["doc_id"], int(r["page_number_in_doc"]) + 1) for r in meta}

    evidence: dict[int, list[dict]] = {}
    for r in qrels:
        if r["score"] <= 0:
            continue
        doc_id, page = page_of[r["corpus_id"]]
        evidence.setdefault(r["query_id"], []).append({"doc_id": doc_id, "page": page, "score": r["score"]})

    english = [q for q in queries if str(q["language"]).lower().startswith("en") and q["query_id"] in evidence]
    rng = random.Random(CFG.seed)
    rng.shuffle(english)
    picked = english[: args.n]

    questions = [
        Question(
            qid=f"vidore-{q['query_id']}",
            collection=COLLECTION,
            question=q["query"],
            answer=q["answer"],
            answer_format="Open",
            doc_ids=[],
            evidence=[{"doc_id": e["doc_id"], "page": e["page"]} for e in evidence[q["query_id"]]],
            tags=tags_for(q["content_type"], len(evidence[q["query_id"]])),
            source="vidore_v3",
            extra={"query_types": q["query_types"], "raw_answers": q["raw_answers"],
                   "relevance": {f"{e['doc_id']}#{e['page']}": e["score"] for e in evidence[q["query_id"]]}},
        )
        for q in picked
    ]
    save_questions(args.out, questions)
    print(f"Saved {len(questions)} English queries -> {args.out}")
    print("Tag counts:", dict(Counter(t for q in questions for t in q.tags)))

    store = PageStore(CFG.pages_dir, COLLECTION)
    store.root.mkdir(parents=True, exist_ok=True)
    cols = ["corpus_id", "doc_id", "page_number_in_doc", "markdown"] + ([] if args.no_images else ["image"])
    for r in corpus.select_columns(cols):
        doc_id, page = r["doc_id"], int(r["page_number_in_doc"]) + 1
        if store.get(doc_id, page):
            continue
        img_rel = f"{safe_name(doc_id)}/p{page:04d}.jpg"
        if not args.no_images:
            (store.root / safe_name(doc_id)).mkdir(parents=True, exist_ok=True)
            r["image"].convert("RGB").save(store.root / img_rel, quality=90)
        text = (r["markdown"] or "").strip()
        store.add(Page(COLLECTION, doc_id, page, img_rel, text, scanned=len(text) < CFG.min_text_chars))
    store.save()
    print(f"Page store: {len(store)} pages over {len(store.doc_ids())} documents in {store.root}")


if __name__ == "__main__":
    main()
