"""Visual page retrieval for B3: ColQwen2 turns every page image into ~700 small vectors,
and a query is matched against them with late interaction (MaxSim).

Indexing runs once, before the answering VLM is loaded (see scripts/index_pages.py): page and
query vectors are written to disk, then ColQwen is unloaded so the VLM gets both GPUs.
At answer time, search is plain numpy on the CPU.

Files per collection, in data/indexes/<collection>/colqwen/:
  shard_00000.npz ...   page vectors, concatenated, with per-page offsets and keys
  queries.npz           query vectors keyed by qid
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import CFG

COLQWEN_MODEL = "vidore/colqwen2-v1.0-hf"
# ColQwen was trained with up to ~768 visual tokens per page; more costs time and memory, not recall.
COLQWEN_MAX_PIXELS = 768 * 28 * 28


def index_dir(collection: str) -> Path:
    return CFG.index_dir / collection / "colqwen"


def maxsim_scores(query: np.ndarray, tokens: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Late-interaction score of one query against every page.

    query: (q, d). tokens: all page vectors stacked, (N, d). offsets: start row of each page, (P,).
    Score of a page = sum over query tokens of the best-matching page token.
    """
    sims = query.astype(np.float32, copy=False) @ tokens.astype(np.float32, copy=False).T      # (q, N)
    best = np.maximum.reduceat(sims, offsets, axis=1)                  # (q, P): max within each page
    return best.sum(axis=0)


class VisualIndex:
    """Page vectors for one collection, loaded from the shards written by scripts/index_pages.py."""

    def __init__(self, keys: list[tuple[str, int]], tokens: np.ndarray, offsets: np.ndarray,
                 queries: dict[str, np.ndarray]):
        self.keys = keys
        self.tokens = tokens
        self.offsets = offsets
        self.queries = queries
        self._doc_pages: dict[str, np.ndarray] = {}
        for i, (d, _) in enumerate(keys):
            self._doc_pages.setdefault(d, []).append(i)
        self._doc_pages = {d: np.array(v) for d, v in self._doc_pages.items()}

    @classmethod
    def load(cls, collection: str) -> "VisualIndex":
        folder = index_dir(collection)
        shards = sorted(folder.glob("shard_*.npz"))
        if not shards:
            raise FileNotFoundError(f"No ColQwen index for '{collection}'. Run scripts/index_pages.py first.")
        keys, toks, offsets, start = [], [], [], 0
        for s in shards:
            z = np.load(s, allow_pickle=False)
            keys += [(d, int(p)) for d, p in json.loads(str(z["keys"]))]
            toks.append(z["tokens"])
            offsets.append(z["offsets"] + start)
            start += len(z["tokens"])
        queries = {}
        qfile = folder / "queries.npz"
        if qfile.exists():
            z = np.load(qfile, allow_pickle=False)
            queries = {k: z[k].astype(np.float32) for k in z.files}
        return cls(keys, np.concatenate(toks).astype(np.float32), np.concatenate(offsets), queries)

    def search(self, qid: str, k: int = CFG.top_k, doc_ids: list[str] | None = None) -> list[tuple[tuple[str, int], float]]:
        if qid not in self.queries:
            raise KeyError(f"No ColQwen query vector for {qid}. Rerun scripts/index_pages.py with its question file.")
        scores = maxsim_scores(self.queries[qid], self.tokens, self.offsets)
        if doc_ids:  # question about named documents: only rank their pages
            rows = np.concatenate([self._doc_pages[d] for d in doc_ids if d in self._doc_pages] or [np.array([], int)])
        else:
            rows = np.arange(len(self.keys))
        if rows.size == 0:
            return []
        order = rows[np.argsort(-scores[rows])[:k]]
        return [(self.keys[i], float(scores[i])) for i in order]


class ColQwen:
    """Thin wrapper over the transformers ColQwen2 model (GPU 0, fp16 on T4)."""

    def __init__(self, model_id: str = COLQWEN_MODEL):
        import torch
        from transformers import ColQwen2ForRetrieval, ColQwen2Processor

        self.torch = torch
        native_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if native_bf16 else torch.float16
        self.model = ColQwen2ForRetrieval.from_pretrained(
            model_id, dtype=dtype, device_map="cuda:0" if torch.cuda.is_available() else "cpu").eval()
        self.processor = ColQwen2Processor.from_pretrained(model_id)

    def _embed(self, batch) -> list[np.ndarray]:
        batch = batch.to(self.model.device)
        with self.torch.inference_mode():
            emb = self.model(**batch).embeddings                       # (B, L, 128), padding rows are 0
            real = batch["attention_mask"].bool()                       # ignore padding rows
            if self.torch.isnan(emb[real]).any() and self.model.dtype == self.torch.float16:
                # Qwen2-VL can overflow in fp16 (T4 has no bf16). fp32 fits for a 2B model: switch for good.
                print("ColQwen: NaN in fp16, switching to fp32", flush=True)
                self.model = self.model.float()
                batch = {k: v.float() if v.is_floating_point() else v for k, v in batch.items()}
                emb = self.model(**batch).embeddings
        mask = batch["attention_mask"].bool()
        return [e[m].float().cpu().numpy().astype(np.float16) for e, m in zip(emb, mask)]

    def embed_images(self, images) -> list[np.ndarray]:
        from .vlm import fit_pixels
        imgs = [fit_pixels(im.convert("RGB"), COLQWEN_MAX_PIXELS) for im in images]
        return self._embed(self.processor(images=imgs, return_tensors="pt"))

    def embed_queries(self, texts: list[str]) -> list[np.ndarray]:
        return self._embed(self.processor(text=texts, return_tensors="pt"))
