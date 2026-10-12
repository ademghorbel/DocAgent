"""One question format and one page format shared by every benchmark and every system.

Pages are numbered from 1 everywhere, so citations match what a human sees in a PDF viewer.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

# Evidence/modality tags used in the results table.
TAGS = ("text", "table", "figure", "multi-page", "unanswerable")
UNANSWERABLE = "Not answerable"


@dataclass
class Question:
    qid: str
    collection: str                 # which page store to search (e.g. "manuals", "vidore_industrial")
    question: str
    answer: str                     # gold answer; UNANSWERABLE when the documents don't contain it
    answer_format: str = "Str"      # Str | Int | Float | List | None | Open (long free-text answer)
    doc_ids: list[str] = field(default_factory=list)   # documents the question is about (empty = whole collection)
    evidence: list[dict] = field(default_factory=list) # [{"doc_id": ..., "page": 3}, ...]
    tags: list[str] = field(default_factory=list)
    source: str = ""                # mmlongbench | vidore_v3 | own
    extra: dict = field(default_factory=dict)

    @property
    def evidence_keys(self) -> set[tuple[str, int]]:
        return {(e["doc_id"], int(e["page"])) for e in self.evidence}


@dataclass
class Page:
    collection: str
    doc_id: str
    page: int                       # 1-based
    image: str                      # path relative to the collection folder
    text: str = ""
    scanned: bool = False           # True when the page had no usable text layer

    @property
    def key(self) -> tuple[str, int]:
        return (self.doc_id, self.page)


def safe_name(name: str) -> str:
    """Folder-safe version of a document id."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:120]


# ---------- jsonl helpers ----------

def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and not line.lstrip().startswith("//")]


def load_questions(path: Path) -> list[Question]:
    return [Question(**row) for row in read_jsonl(path)]


def save_questions(path: Path, questions: Iterable[Question]) -> int:
    return write_jsonl(path, (asdict(q) for q in questions))


# ---------- page store ----------

class PageStore:
    """Rendered pages for one collection: <pages_dir>/<collection>/pages.jsonl + image files."""

    def __init__(self, pages_dir: Path, collection: str):
        self.root = Path(pages_dir) / collection
        self.collection = collection
        self.pages: list[Page] = []
        self._by_key: dict[tuple[str, int], Page] = {}
        manifest = self.root / "pages.jsonl"
        if manifest.exists():
            for row in read_jsonl(manifest):
                self._add(Page(**row))

    def _add(self, page: Page):
        self.pages.append(page)
        self._by_key[page.key] = page

    def add(self, page: Page):
        self._add(page)

    def save(self):
        self.pages.sort(key=lambda p: (p.doc_id, p.page))
        write_jsonl(self.root / "pages.jsonl", (asdict(p) for p in self.pages))

    def get(self, doc_id: str, page: int) -> Page | None:
        return self._by_key.get((doc_id, int(page)))

    def image_path(self, page: Page) -> Path:
        return self.root / page.image

    def doc_pages(self, doc_id: str) -> list[Page]:
        return sorted((p for p in self.pages if p.doc_id == doc_id), key=lambda p: p.page)

    def doc_ids(self) -> list[str]:
        return sorted({p.doc_id for p in self.pages})

    def __len__(self):
        return len(self.pages)
