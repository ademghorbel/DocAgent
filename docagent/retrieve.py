"""Text retrieval for the B1 baseline: page text -> chunks -> bge embeddings -> FAISS.

Every chunk remembers its (doc_id, page), so B1 can cite pages and we can measure page recall@k
the same way we will for ColQwen in week 2.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import CFG
from .data import PageStore

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def encode_docs(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


class BGEEmbedder:
    def __init__(self, model_name: str = CFG.embed_model, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() and "m3" not in model_name.lower() else ""

    def encode_docs(self, texts):
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 500).astype("float32")

    def encode_query(self, text):
        return self.model.encode([self.prefix + text], normalize_embeddings=True).astype("float32")[0]


def chunk_text(text: str, size: int = CFG.chunk_chars, overlap: int = CFG.chunk_overlap) -> list[str]:
    """Split on paragraph boundaries, then pack paragraphs into chunks of about `size` characters."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    paras = [p.strip() for p in text.split("\n\n") if p.strip()] or [text]
    pieces = []
    for p in paras:
        if len(p) > size:  # very long paragraphs (e.g. tables flattened to text) get hard-split
            pieces.extend(p[i:i + size] for i in range(0, len(p), size - overlap))
        else:
            pieces.append(p)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 2 > size:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n\n" + p if overlap else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


@dataclass
class Hit:
    doc_id: str
    page: int
    score: float
    text: str


class TextIndex:
    def __init__(self, chunks: list[dict], vectors: np.ndarray, embedder: Embedder):
        self.chunks = chunks          # [{"doc_id", "page", "text"}]
        self.vectors = vectors
        self.embedder = embedder
        import faiss
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self._doc_rows: dict[str, np.ndarray] = {}
        for i, c in enumerate(chunks):
            self._doc_rows.setdefault(c["doc_id"], []).append(i)
        self._doc_rows = {d: np.array(r) for d, r in self._doc_rows.items()}

    # ----- build / load -----
    @classmethod
    def build(cls, store: PageStore, embedder: Embedder) -> "TextIndex":
        chunks = [{"doc_id": p.doc_id, "page": p.page, "text": c}
                  for p in store.pages for c in chunk_text(p.text)]
        if not chunks:
            raise ValueError(f"No text in collection '{store.collection}'. Scanned PDFs? Re-ingest with --ocr.")
        vectors = embedder.encode_docs([c["text"] for c in chunks])
        return cls(chunks, vectors, embedder)

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        with (folder / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, folder: Path, embedder: Embedder) -> "TextIndex":
        with (folder / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        return cls(chunks, np.load(folder / "vectors.npy"), embedder)

    @classmethod
    def for_collection(cls, store: PageStore, embedder: Embedder, rebuild: bool = False) -> "TextIndex":
        folder = CFG.index_dir / store.collection / "text"
        if folder.exists() and not rebuild:
            return cls.load(folder, embedder)
        idx = cls.build(store, embedder)
        idx.save(folder)
        return idx

    # ----- search -----
    def search(self, query: str, k: int = CFG.top_k, doc_ids: list[str] | None = None) -> list[Hit]:
        q = self.embedder.encode_query(query)
        if doc_ids:  # restrict to some documents: brute force over their rows (small)
            rows = np.concatenate([self._doc_rows[d] for d in doc_ids if d in self._doc_rows] or [np.array([], int)])
            if rows.size == 0:
                return []
            scores = self.vectors[rows] @ q
            order = np.argsort(-scores)[:k]
            ids, sims = rows[order], scores[order]
        else:
            sims, ids = self.index.search(q[None, :], k)
            ids, sims = ids[0], sims[0]
        return [Hit(self.chunks[i]["doc_id"], self.chunks[i]["page"], float(s), self.chunks[i]["text"])
                for i, s in zip(ids, sims) if i >= 0]


def unique_pages(hits: list[Hit]) -> list[tuple[str, int]]:
    """Pages in rank order, without duplicates (several chunks can come from one page)."""
    seen, out = set(), []
    for h in hits:
        key = (h.doc_id, h.page)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out
