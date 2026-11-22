"""Fault catalog: every fault, alarm and error code of every manual, in one table.

The VLM reads each manual's fault pages once and writes one row per code (device, code, name,
causes, remedy, page). Afterwards:
  - an alert or question with a known code is answered straight from the table, with the page;
  - the table can be browsed and searched across all brands;
  - anything else (symptoms, settings, comparisons) still goes to the agent.

Devices are also how questions are routed: "ACS580 fault 2310" only searches the ABB manual.

    python scripts/build_catalog.py            # builds rows for manuals not catalogued yet
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .config import CFG, _env
from .data import PageStore

CATALOG_PATH = Path(_env("DOCAGENT_CATALOG", CFG.data_dir / "catalog" / "faults.jsonl"))
MAX_FAULT_PAGES = _env("DOCAGENT_CATALOG_PAGES", 30, int)   # fault pages read per manual at most
MIN_KEYWORD_HITS = 4
MAX_GAP = 2                # pages filled in between two fault pages

_FAULT_WORDS = re.compile(r"\b(fault|alarm|error|trip|warning|cause|remed|troubleshoot|diagnos|corrective|"
                          r"clear(?:ing)?|reset)\w*", re.I)
# Words in file names that are not part of a device name.
_FILLER = {"programming", "firmware", "io", "easybook", "easy", "book", "manual", "user", "guide",
           "installation", "datasheet", "operating", "instructions", "en", "pdf", "reference", "hardware"}

EXTRACT_PROMPT = """Above is one page{text_note} from the manual of the {device}.
If this page lists fault, alarm, warning or error codes, write one line per code, exactly:
CODE | NAME | PROBABLE CAUSES | REMEDY | CLEARING
- CODE exactly as printed (for example OCF, 2310, E1, A-12).
- Copy the words of the page; do not add causes or remedies the page does not print.
- CLEARING: how the fault is cleared or reset, including any remark, note or key sequence the
  page gives for it (for example "press ENT twice").
- Separate several causes or remedies with ;
- Leave a field empty if the page does not give it, but keep the | separators.
- No header line, no other text.
If the page lists no codes, reply exactly: NONE"""


@dataclass
class Fault:
    device: str
    doc_id: str
    code: str
    name: str = ""
    causes: str = ""
    remedy: str = ""
    pages: list[int] = field(default_factory=list)   # best page (the fault-tracing entry) first
    clearing: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.doc_id, self.code.upper())


# ---------- devices ----------

def _stem_tokens(doc_id: str) -> list[str]:
    return [t for t in re.split(r"[_\-\s.]+", Path(doc_id).stem.lower()) if t]


def device_name(doc_id: str) -> str:
    """'schneider_atv320_programming.pdf' -> 'Schneider ATV320'."""
    toks = _stem_tokens(doc_id)
    if not toks:
        return doc_id
    brand = toks[0].upper() if len(toks[0]) <= 3 else toks[0].capitalize()
    model = [t.upper() for t in toks[1:] if t not in _FILLER]
    return " ".join([brand] + model)


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def route(question: str, doc_ids: list[str]) -> list[str]:
    """Manuals the question is about, from the device names it mentions; [] means all manuals.

    A model name ('ACS580', 'S7-1200', 'SCALA2') picks its manual; a brand alone ('Grundfos') picks
    that brand's manuals. Spaces, dashes and case are ignored, so 'acs 580' matches too.
    """
    q = _squash(question)
    by_model = [d for d in doc_ids
                if any(len(t) >= 3 and re.search(r"\d", t) and t in q for t in _stem_tokens(d))]
    if by_model:
        return by_model
    words = set(re.findall(r"[a-z]+", question.lower()))
    return [d for d in doc_ids if _stem_tokens(d) and len(_stem_tokens(d)[0]) >= 3
            and _stem_tokens(d)[0] in words]


# ---------- finding and reading fault pages ----------

def fault_pages(store: PageStore, doc_id: str, max_pages: int = MAX_FAULT_PAGES) -> list[int]:
    """Pages of a manual that look like fault tables, by fault words in the text layer, in page order."""
    scored = [(len(_FAULT_WORDS.findall(p.text or "")), p.page) for p in store.doc_pages(doc_id)]
    best = sorted(p for _, p in sorted((s for s in scored if s[0] >= MIN_KEYWORD_HITS),
                                       key=lambda s: -s[0])[:max_pages])
    # A fault chapter is a run of pages; a page in it with few keywords (a dense code table) would be
    # skipped. Fill gaps of up to MAX_GAP pages between chosen pages.
    have = {p.page for p in store.doc_pages(doc_id)}
    filled = set(best)
    for a, b in zip(best, best[1:]):
        if 1 < b - a <= MAX_GAP + 1:
            filled.update(p for p in range(a + 1, b) if p in have)
    return sorted(filled)


_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-.]{0,11}$")
_FIGURE_ID = re.compile(r"^[A-Za-z]{1,3}\d{6,}$")
_HEADER = {"code", "fault code", "fault", "alarm", "error", "error code", "alarm code", "warning", "id"}


def parse_rows(text: str, device: str, doc_id: str, page: int) -> list[Fault]:
    out = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip(" `*\t") for c in line.strip().strip("|").split("|")]
        cells += [""] * (5 - len(cells))
        code = cells[0].lstrip("-• ").strip()
        if (not code or code.lower() in _HEADER or not _CODE.match(code) or set(code) <= set("-.")
                or _FIGURE_ID.match(code)):
            continue
        out.append(Fault(device, doc_id, code, cells[1], cells[2], cells[3], [page],
                         " | ".join(c for c in cells[4:] if c)))
    return out


# ---------- grounding: keep only what the page actually prints ----------

_STOP = {"this", "that", "with", "from", "have", "been", "were", "will", "when", "then", "than", "check",
         "each", "into", "some", "such", "also", "only", "more", "other", "page", "below", "above"}
GROUND_MIN = 0.6   # share of an item's content words that must be on the page


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP]


def grounded(item: str, page_text: str) -> bool:
    """True when most content words of a cause/remedy item are printed on the page.
    Pages with no text layer (scans) can't be checked, so their items are kept."""
    words = _words(item)
    if not page_text.strip() or not words:
        return True
    vocab = set(_words(page_text))
    return sum(w in vocab for w in words) / len(words) >= GROUND_MIN


def ground_field(value: str, page_text: str) -> str:
    """Drop the ;-separated items the page doesn't print (the VLM sometimes fills in generic causes)."""
    return "; ".join(i.strip() for i in value.split(";") if i.strip() and grounded(i, page_text))


_IS_CLEARING = re.compile(r"(detected error|fault|it)\b.*\b(clear|reset)", re.I)
_IS_ACTION = re.compile(r"^(check|replace|add|reduce|increase|decrease|contact|turn|verify|set|disable|"
                        r"restore|reboot|load|apply|reinsert|confirm|re-?calibrate|wait|use|refer|"
                        r"test|connect|clean|repair|adjust|prime|switch|ensure|remove|retry|try)\b", re.I)


def _items(text: str) -> list[str]:
    return [i.strip(" |") for i in re.split(r"[;|]", text) if i.strip(" |")]


def tidy(row: Fault) -> Fault:
    """Move each item to its column: actions out of causes, clearing sentences out of remedy."""
    causes, remedy, clearing = [], [], _items(row.clearing)
    for i in _items(row.causes):
        (remedy if _IS_ACTION.match(i) else causes).append(i)
    for i in _items(row.remedy):
        (clearing if _IS_CLEARING.search(i) and not _IS_ACTION.match(i) else remedy).append(i)
    row.causes, row.remedy = "; ".join(causes), "; ".join(dict.fromkeys(remedy))
    row.clearing = "; ".join(dict.fromkeys(clearing))
    return row


def ground(row: Fault, page_text: str) -> Fault:
    tidy(row)
    for f in ("causes", "remedy", "clearing"):
        setattr(row, f, ground_field(getattr(row, f), page_text))
    return row


def _weight(r: Fault) -> int:
    return len(r.causes) + len(r.remedy) + len(r.clearing)


def merge(rows: list[Fault]) -> list[Fault]:
    """One row per (manual, code): longest text per field, all pages kept.

    A code often shows up on several pages (a parameter table that names it, then its entry in the
    fault-tracing chapter). The page that gave the most causes/remedy text comes first: that is the
    page shown as the source."""
    merged: dict[tuple[str, str], Fault] = {}
    best: dict[tuple[str, str], tuple[int, int]] = {}
    for r in rows:
        m = merged.get(r.key)
        if m is None:
            merged[r.key] = Fault(**{**asdict(r), "pages": list(r.pages)})
            best[r.key] = (_weight(r), r.pages[0] if r.pages else 0)
            continue
        for f in ("name", "causes", "remedy", "clearing"):
            if len(getattr(r, f)) > len(getattr(m, f)):
                setattr(m, f, getattr(r, f))
        if r.pages and _weight(r) > best[r.key][0]:
            best[r.key] = (_weight(r), r.pages[0])
        m.pages = sorted(set(m.pages) | set(r.pages))
    for key, m in merged.items():
        top = best[key][1]
        if top in m.pages:
            m.pages = [top] + [p for p in m.pages if p != top]
    return list(merged.values())


# ---------- the catalog ----------

class Catalog:
    """faults.jsonl, plus done.json: the manuals already read (also those with no fault codes)."""

    def __init__(self, path: Path = CATALOG_PATH):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.faults: list[Fault] = []
        self.done: set[str] = set()
        seed = CFG.data_dir / "catalog" / "faults.jsonl"   # committed copy, if the path points elsewhere
        src = self.path if self.path.exists() else seed if seed.exists() else None
        if src:
            with src.open(encoding="utf-8") as f:
                self.faults = [Fault(**json.loads(line)) for line in f if line.strip()]
            done = src.with_name("done.json")
            self.done = set(json.loads(done.read_text())) if done.exists() else {x.doc_id for x in self.faults}

    def save(self):
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as f:
                for x in sorted(self.faults, key=lambda x: (x.doc_id, x.code)):
                    f.write(json.dumps(asdict(x), ensure_ascii=False) + "\n")
            self.path.with_name("done.json").write_text(json.dumps(sorted(self.done)))

    def add(self, rows: list[Fault], doc_id: str):
        with self._lock:   # the demo builds in the background while answering
            self.faults = merge([x for x in self.faults if x.doc_id != doc_id] + rows)
            self.done.add(doc_id)

    def reground(self, store: PageStore) -> int:
        """Re-check rows built before grounding existed, against their pages' text; no model calls.
        Returns how many rows changed."""
        changed = 0
        with self._lock:
            n = len(self.faults)
            self.faults = [x for x in self.faults if not _FIGURE_ID.match(x.code)]
            changed += n - len(self.faults)
            for x in self.faults:
                texts = {p: (store.get(x.doc_id, p).text or "") for p in x.pages if store.get(x.doc_id, p)}
                if not texts:
                    continue
                before = (x.causes, x.remedy, x.clearing, list(x.pages))
                ground(x, " ".join(texts.values()))
                # source page first: the one printing most of the remaining causes/remedy words
                words = set(_words(f"{x.causes} {x.remedy} {x.clearing}"))
                x.pages.sort(key=lambda p: -len(words & set(_words(texts.get(p, "")))))
                changed += before != (x.causes, x.remedy, x.clearing, x.pages)
        return changed

    def __len__(self):
        return len(self.faults)

    def find_codes(self, question: str, doc_ids: list[str] | None = None) -> list[Fault]:
        """Faults whose code appears in the question as a whole token.

        Short codes are only trusted when they can't be ordinary text: a pure number needs 3+ digits
        or a word like 'fault' right before it ('line 2' is not fault 2); letters-only codes need 3+.
        """
        hits = []
        for x in self.faults:
            if doc_ids and x.doc_id not in doc_ids:
                continue
            c = re.escape(x.code)
            if x.code.isdigit() and len(x.code) < 3:
                pat = rf"\b(?:fault|alarm|error|warning|code|trip)\s*(?:code\s*)?#?\s*{c}(?![\w.])"
                flags = re.I
            elif x.code.isalpha() and len(x.code) < 3:
                continue
            else:
                # codes with a digit can't be an English word, so case doesn't matter ('scf1');
                # letter codes match as printed, or typed in capitals ('PHF' for 'phf')
                pat = rf"(?<![A-Za-z0-9]){c}(?![A-Za-z0-9])"
                flags = re.I
            m = re.search(pat, question, flags)
            if m and (re.search(r"\d", x.code) or m.group(0) == x.code or m.group(0).isupper()):
                hits.append(x)
        return hits

    def search(self, text: str = "", doc_ids: list[str] | None = None) -> list[Fault]:
        words = text.lower().split()
        return [x for x in self.faults
                if (not doc_ids or x.doc_id in doc_ids)
                and all(w in f"{x.device} {x.code} {x.name} {x.causes} {x.remedy} {x.clearing}".lower()
                        for w in words)]


def build(store: PageStore, vlm, catalog: Catalog, doc_ids: list[str] | None = None,
          progress: Callable[[str], None] = print, save_every_doc: bool = True) -> Catalog:
    """Read the fault pages of each manual not yet in the catalog and add its codes."""
    from PIL import Image

    from .prompts import page_label

    todo = [d for d in (doc_ids or store.doc_ids()) if d not in catalog.done]
    for n, doc in enumerate(todo, 1):
        device, pages, rows = device_name(doc), fault_pages(store, doc), []
        for i, pno in enumerate(pages, 1):
            page = store.get(doc, pno)
            text = (page.text or "").strip()[:3000]
            prompt = (f"Text layer of this page:\n{text}\n\n" if text else "") + EXTRACT_PROMPT.format(
                device=device, text_note=" and its text layer" if text else "")
            g = vlm.generate(prompt, images=[(page_label(doc, pno), Image.open(store.image_path(page)))],
                             max_new_tokens=700)
            found = [ground(r, page.text or "") for r in parse_rows(g.text, device, doc, pno)]
            rows += found
            progress(f"{device} ({n}/{len(todo)}): page {pno} ({i}/{len(pages)}), {len(found)} codes")
        catalog.add(rows, doc)
        progress(f"{device}: {len(merge(rows))} codes from {len(pages)} pages")
        if save_every_doc:
            catalog.save()
    return catalog
