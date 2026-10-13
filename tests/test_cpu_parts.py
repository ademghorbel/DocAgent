"""CPU-only tests: no GPU, no model downloads. Run with `pytest -q`."""
import numpy as np
import pymupdf
import pytest

from docagent.config import CFG
from docagent.data import UNANSWERABLE, PageStore, Question
from docagent.ingest import ingest_pdf
from docagent.prompts import parse_reply
from docagent.retrieve import TextIndex, chunk_text, unique_pages
from docagent.scoring import citation_scores, recall_at_k, rule_score, summary_table


class HashEmbedder:
    """Bag-of-words hashing embedder so retrieval can be tested without downloading bge."""
    dim = 256

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for w in text.lower().split():
            v[hash(w.strip(".,:;?")) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text):
        return self._vec(text)


@pytest.fixture
def store(tmp_path):
    pdf = tmp_path / "pump_A.pdf"
    doc = pymupdf.open()
    pages = [
        "Pump P-12 installation guide. Mount the pump on a level concrete base.",
        "Troubleshooting table. Error E-47: discharge pressure too high. Cause: closed outlet valve. "
        "Remedy: open the outlet valve and check the pressure sensor.",
        "",  # blank page stands in for a scanned page (no text layer)
        "Technical data. Maximum operating temperature 90 C. Rated current 4.2 A.",
    ]
    for t in pages:
        p = doc.new_page()
        if t:
            p.insert_textbox(pymupdf.Rect(50, 50, 550, 800), t, fontsize=11)
    doc.save(pdf)
    s = PageStore(tmp_path / "pages", "manuals")
    s.root.mkdir(parents=True)
    stats = ingest_pdf(pdf, s, dpi=50)
    s.save()
    assert stats["pages"] == 4 and stats["scanned"] == 1
    return s


def test_ingest_roundtrip(store):
    reloaded = PageStore(store.root.parent, "manuals")
    assert len(reloaded) == 4
    p2 = reloaded.get("pump_A.pdf", 2)
    assert "E-47" in p2.text and reloaded.image_path(p2).exists()
    assert reloaded.get("pump_A.pdf", 3).scanned


def test_ingest_is_idempotent(store, tmp_path):
    stats = ingest_pdf(tmp_path / "pump_A.pdf", store, dpi=50)
    assert stats["pages"] == 4 and len(store) == 4


def test_chunking():
    assert chunk_text("") == []
    assert chunk_text("short") == ["short"]
    long = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(10))
    chunks = chunk_text(long, size=1000, overlap=100)
    assert len(chunks) > 1 and all(len(c) <= 1000 + 400 for c in chunks)
    huge = "y" * 5000  # one giant paragraph, e.g. a flattened table
    assert len(chunk_text(huge, size=1000, overlap=100)) >= 5


def test_text_retrieval_finds_error_code_page(store):
    idx = TextIndex.build(store, HashEmbedder())
    hits = idx.search("what does error E-47 mean, discharge pressure too high", k=2)
    assert unique_pages(hits)[0] == ("pump_A.pdf", 2)
    filtered = idx.search("maximum operating temperature", k=3, doc_ids=["pump_A.pdf"])
    assert filtered[0].page == 4
    assert idx.search("anything", doc_ids=["missing.pdf"]) == []


def test_index_save_load(store, tmp_path):
    idx = TextIndex.build(store, HashEmbedder())
    idx.save(tmp_path / "idx")
    again = TextIndex.load(tmp_path / "idx", HashEmbedder())
    assert len(again.chunks) == len(idx.chunks)


def test_parse_reply():
    ans, cited = parse_reply("Answer: 90 C\nPages: [pump_A.pdf p.4], [pump_A.pdf p.4], [drive.pdf p.12]")
    assert ans == "90 C" and cited == [("pump_A.pdf", 4), ("drive.pdf", 12)]
    ans, cited = parse_reply("The answer is 4.2 A")
    assert ans == "The answer is 4.2 A" and cited == []
    _, cited = parse_reply("Answer: x\nPages: none")
    assert cited == []


def q(answer, fmt="Str"):
    return Question("t", "c", "?", answer, fmt)


@pytest.mark.parametrize("gold,fmt,pred,expected", [
    ("90", "Int", "The maximum is 90 °C", 1.0),
    ("90", "Int", "85", 0.0),
    ("4.2", "Float", "4,2 A", 1.0),
    ("21%", "Float", "0.21", 1.0),
    ("1,200", "Int", "1200 rpm", 1.0),
    ("Berlin School of Experimental Psychology", "Str", "the Berlin School of Experimental Psychology", 1.0),
    ("closed outlet valve", "Str", "A closed outlet valve", 1.0),
    ("closed outlet valve", "Str", "worn impeller", 0.0),
    ("['pump', 'valve']", "List", "valve, pump", 1.0),
    ("['pump', 'valve']", "List", "pump", 0.0),
    (UNANSWERABLE, "None", "Not answerable", 1.0),
    (UNANSWERABLE, "None", "The document does not contain this information.", 1.0),
    (UNANSWERABLE, "None", "24 months", 0.0),
    ("90", "Int", "Not answerable", 0.0),
])
def test_rule_score(gold, fmt, pred, expected):
    assert rule_score(q(gold, fmt), pred) == expected


def test_open_answers_need_judge():
    assert rule_score(q("Release the deadman control, then ...", "Open"), "Some steps") is None


def test_retrieval_and_citation_metrics():
    ev = {("a", 1), ("a", 2)}
    assert recall_at_k([("a", 1), ("b", 9)], ev, 5) == 0.5
    assert recall_at_k([("b", 9), ("a", 1)], ev, 1) == 0.0
    assert recall_at_k([], set(), 5) is None
    assert citation_scores([("a", 1)], ev) == (1.0, 0.5)
    assert citation_scores([], ev) == (0.0, 0.0)


def test_summary_table():
    qs = {"q1": Question("q1", "c", "?", "1", tags=["table"]),
          "q2": Question("q2", "c", "?", "2", tags=["text", "multi-page"])}
    rows = [
        {"system": "B1_text_rag", "qid": "q1", "score": 1.0, "recall_at_k": 1.0, "cite_precision": 1.0, "latency_s": 2.0},
        {"system": "B1_text_rag", "qid": "q2", "score": 0.0, "recall_at_k": 0.5, "cite_precision": 0.0, "latency_s": 4.0},
        {"system": "B2_direct_vlm", "qid": "q1", "skipped": "needs exactly one doc_id"},
    ]
    t = summary_table(rows, qs)
    assert "| B1_text_rag | 50.0 (2) | 0.0 (1) | 100.0 (1) |" in t
    assert "B2_direct_vlm" not in t


def test_summary_counts_retrieval_on_unscored_rows():
    qs = {"q1": Question("q1", "c", "?", "long open answer", "Open", tags=["text"])}
    rows = [{"system": "B1_text_rag", "qid": "q1", "score": None, "recall_at_k": 0.5,
             "cite_precision": 1.0, "latency_s": 3.0}]
    t = summary_table(rows, qs)
    assert "| B1_text_rag | – |" in t and "| 0.50 | 1.00 | 3.0 |" in t
