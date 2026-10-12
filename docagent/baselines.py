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


def run_b1(q: Question, index: TextIndex, vlm: VLM, k: int = CFG.top_k) -> dict:
    hits = index.search(q.question, k=k, doc_ids=q.doc_ids or None)
    context = "\n\n".join(f"{page_label(h.doc_id, h.page)}\n{h.text}" for h in hits)
    gen = vlm.generate(answer_prompt(q.question, q.answer_format, context), system=SYSTEM)
    return _record("B1_text_rag", q, gen, unique_pages(hits))


def run_b2(q: Question, store: PageStore, vlm: VLM, n_pages: int = CFG.direct_vlm_pages) -> dict:
    # Direct VLM only makes sense when the question names one document. Corpus-wide questions
    # (ViDoRe) are skipped rather than handed the gold document, which would leak the answer location.
    if len(q.doc_ids) != 1:
        return _record("B2_direct_vlm", q, None, [], skipped="needs exactly one doc_id")
    pages = store.doc_pages(q.doc_ids[0])[:n_pages]
    images = [(page_label(p.doc_id, p.page), Image.open(store.image_path(p))) for p in pages]
    gen = vlm.generate(answer_prompt(q.question, q.answer_format), images=images, system=SYSTEM)
    return _record("B2_direct_vlm", q, gen, [p.key for p in pages])


SYSTEMS = {"B1": run_b1, "B2": run_b2}
