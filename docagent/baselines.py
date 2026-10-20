"""Baselines.

B1 text RAG: page text -> top-k chunks (bge + FAISS) -> VLM answers from text only.
B2 direct VLM: the first N page images of the document -> VLM, no retrieval.
B3 multimodal RAG: ColQwen top-k page images -> VLM, single shot (week 2).

All return the same record shape, which run_eval.py scores.
"""
from __future__ import annotations

from PIL import Image

from .config import CFG
from .data import PageStore, Question
from .prompts import SYSTEM, answer_prompt, page_label, parse_reply
from .retrieve import TextIndex, unique_pages
from .vlm import VLM


def _record(system: str, q: Question, gen, retrieved: list[tuple[str, int]], skipped: str = "") -> dict:
    answer, cited = parse_reply(gen.text) if gen else ("", [])
    return {
        "system": system,
        "qid": q.qid,
        "prediction": answer,
        "raw": gen.text if gen else "",
        "cited": [list(c) for c in cited],
        "retrieved": [list(r) for r in retrieved],
        "latency_s": round(gen.latency_s, 2) if gen else None,
        "input_tokens": gen.input_tokens if gen else None,
        "peak_gpu_gb": gen.peak_gpu_gb if gen else None,
        "skipped": skipped,
    }


def select_pages(hits, k: int, max_chunks_per_page: int = 2):
    """Keep chunks from the first k distinct pages (at most 2 chunks per page).

    Several of the top chunks often come from the same page, so "top-5 chunks" can mean only
    2-3 pages. Retrieval is compared page-for-page with ColQwen in week 2, so B1 must also
    hand over k distinct pages.
    """
    pages, per_page, kept = [], {}, []
    for h in hits:
        key = (h.doc_id, h.page)
        if key not in per_page:
            if len(pages) == k:
                continue
            pages.append(key)
            per_page[key] = 0
        if per_page[key] < max_chunks_per_page:
            per_page[key] += 1
            kept.append(h)
    return kept, pages


def run_b1(q: Question, index: TextIndex, vlm: VLM, k: int = CFG.top_k) -> dict:
    hits = index.search(q.question, k=k * 4, doc_ids=q.doc_ids or None)
    hits, _ = select_pages(hits, k)
    context = "\n\n".join(f"{page_label(h.doc_id, h.page)}\n{h.text}" for h in hits)
    gen = vlm.generate(answer_prompt(q.question, q.answer_format, context), system=SYSTEM)
    return _record("B1_text_rag", q, gen, unique_pages(hits))


def _ask_with_images(q: Question, store: PageStore, pages, vlm: VLM, tag: str):
    """Send page images to the VLM. On out-of-memory, halve the pages and retry (4 -> 2 -> 1).
    Returns (gen, pages actually sent), so the results table stays honest."""
    n = len(pages)
    while True:
        used = pages[:n]
        images = [(page_label(p.doc_id, p.page), Image.open(store.image_path(p))) for p in used]
        try:
            return vlm.generate(answer_prompt(q.question, q.answer_format), images=images, system=SYSTEM), used
        except Exception as e:  # noqa: BLE001
            if "out of memory" not in str(e).lower() or n == 1:
                raise
            vlm.free_memory()
            n = max(1, n // 2)
            print(f"  {q.qid}: out of GPU memory, retrying {tag} with {n} page(s)")


def run_b2(q: Question, store: PageStore, vlm: VLM, n_pages: int = CFG.direct_vlm_pages) -> dict:
    # Direct VLM only makes sense when the question names one document. Corpus-wide questions
    # (ViDoRe) are skipped rather than handed the gold document, which would leak the answer location.
    if len(q.doc_ids) != 1:
        return _record("B2_direct_vlm", q, None, [], skipped="needs exactly one doc_id")
    gen, used = _ask_with_images(q, store, store.doc_pages(q.doc_ids[0])[:n_pages], vlm, "B2")
    row = _record("B2_direct_vlm", q, gen, [p.key for p in used])
    row["pages_used"] = len(used)
    return row


def run_b3(q: Question, store: PageStore, vindex, vlm: VLM, k: int = CFG.top_k,
           n_pages: int = CFG.direct_vlm_pages) -> dict:
    """B3 multimodal RAG: ColQwen ranks page images, the top pages go to the VLM in one shot.
    Recall is measured on the top k; the VLM sees the top n_pages (same image budget as B2)."""
    ranked = [key for key, _ in vindex.search(q.qid, k=k, doc_ids=q.doc_ids or None)]
    pages = [p for p in (store.get(d, pg) for d, pg in ranked[:n_pages]) if p is not None]
    gen, used = _ask_with_images(q, store, pages, vlm, "B3")
    row = _record("B3_colqwen_rag", q, gen, ranked)
    row["pages_used"] = len(used)
    return row


SYSTEMS = {"B1": run_b1, "B2": run_b2, "B3": run_b3}
