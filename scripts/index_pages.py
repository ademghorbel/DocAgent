"""Build the ColQwen page index (B3) for every collection used by the given question files.

    python scripts/index_pages.py --questions data/questions/*.jsonl

Pages are embedded in shards of 256, so a session that stops can just rerun this: finished
shards are kept. Query vectors are recomputed every time (a few seconds). If a page store has
changed size since the index was built, that collection is re-indexed from scratch.
About 4-6 pages/s on one T4, so ~25 minutes for the ~7,400 pages of week 1.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.config import CFG  # noqa: E402
from docagent.data import PageStore, load_questions  # noqa: E402
from docagent.visual import COLQWEN_MODEL, ColQwen, index_dir  # noqa: E402

SHARD = 256


def index_collection(model: ColQwen, store: PageStore, batch: int):
    folder = index_dir(store.collection)
    meta_path = folder / "meta.json"
    meta = {"model": COLQWEN_MODEL, "n_pages": len(store)}
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        print(f"{store.collection}: page store changed since last index, rebuilding")
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta))

    pages = sorted(store.pages, key=lambda p: (p.doc_id, p.page))
    n_shards = (len(pages) + SHARD - 1) // SHARD
    start = time.time()
    for s in range(n_shards):
        out = folder / f"shard_{s:05d}.npz"
        if out.exists():
            continue
        chunk = pages[s * SHARD:(s + 1) * SHARD]
        vecs = []
        for i in range(0, len(chunk), batch):
            imgs = [Image.open(store.image_path(p)) for p in chunk[i:i + batch]]
            vecs += model.embed_images(imgs)
        offsets = np.cumsum([0] + [len(v) for v in vecs[:-1]])
        np.savez(out, tokens=np.concatenate(vecs), offsets=offsets,
                 keys=json.dumps([[p.doc_id, p.page] for p in chunk]))
        done = min((s + 1) * SHARD, len(pages))
        print(f"  {store.collection}: {done}/{len(pages)} pages ({done / (time.time() - start):.1f} pages/s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", nargs="+", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    by_col: dict[str, list] = {}
    for f in args.questions:
        for q in load_questions(f):
            by_col.setdefault(q.collection, []).append(q)

    print(f"Loading {COLQWEN_MODEL}...", flush=True)
    model = ColQwen()
    for col, qs in by_col.items():
        store = PageStore(CFG.pages_dir, col)
        if not len(store):
            print(f"{col}: no pages, skipped")
            continue
        print(f"{col}: {len(store)} pages, {len(qs)} questions", flush=True)
        index_collection(model, store, args.batch)
        qvecs = {}
        for i in range(0, len(qs), 16):
            part = qs[i:i + 16]
            qvecs.update({q.qid: v for q, v in zip(part, model.embed_queries([q.question for q in part]))})
        np.savez(index_dir(col) / "queries.npz", **qvecs)
    print("ColQwen index ready.")


if __name__ == "__main__":
    main()
