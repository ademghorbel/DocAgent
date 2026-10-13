"""Week-1 baselines.

B1 text RAG: page text -> top-k chunks (bge + FAISS) -> VLM answers from text only.
B2 direct VLM: the first N page images of the document -> VLM, no retrieval.

Both return the same record shape, which run_eval.py scores.
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


def run_b2(q: Question, store: PageStore, vlm: VLM, n_pages: int = CFG.direct_vlm_pages) -> dict:
    # Direct VLM only makes sense when the question names one document. Corpus-wide questions
    # (ViDoRe) are skipped rather than handed the gold document, which would leak the answer location.
    if len(q.doc_ids) != 1:
        return _record("B2_direct_vlm", q, None, [], skipped="needs exactly one doc_id")
    # If the GPU runs out of memory, halve the number of pages and retry (8 -> 4 -> 2 -> 1).
    # The pages actually used are recorded, so the results table stays honest.
    n = n_pages
    while True:
        pages = store.doc_pages(q.doc_ids[0])[:n]
        images = [(page_label(p.doc_id, p.page), Image.open(store.image_path(p))) for p in pages]
        try:
            gen = vlm.generate(answer_prompt(q.question, q.answer_format), images=images, system=SYSTEM)
            break
        except Exception as e:  # noqa: BLE001
            if "out of memory" not in str(e).lower() or n == 1:
                raise
            vlm.free_memory()
            n = max(1, n // 2)
            print(f"  {q.qid}: out of GPU memory, retrying B2 with {n} page(s)")
    row = _record("B2_direct_vlm", q, gen, [p.key for p in pages])
    row["pages_used"] = len(pages)
    return row


SYSTEMS = {"B1": run_b1, "B2": run_b2}
