"""End-to-end B1/B2 + scoring with a fake VLM (no GPU, no downloads)."""
import sys
from pathlib import Path

from docagent.baselines import run_b1, run_b2
from docagent.data import Question
from docagent.retrieve import TextIndex
from docagent.vlm import GenResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from run_eval import score_row  # noqa: E402
from test_cpu_parts import HashEmbedder, store  # noqa: E402,F401  (reuse the fixture)


class FakeVLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def generate(self, prompt, images=None, system=None, max_new_tokens=256):
        self.calls.append({"prompt": prompt, "n_images": len(images or [])})
        return GenResult(self.reply, 0.5, 100, 0.0)


def test_b1_and_b2_end_to_end(store):  # noqa: F811
    q = Question("q1", "manuals", "What does error E-47 mean?", "discharge pressure too high", "Str",
                 doc_ids=["pump_A.pdf"], evidence=[{"doc_id": "pump_A.pdf", "page": 2}], tags=["table"])
    vlm = FakeVLM("Answer: Discharge pressure too high\nPages: [pump_A.pdf p.2]")

    r1 = score_row(run_b1(q, TextIndex.build(store, HashEmbedder()), vlm, k=3), q, "rule", vlm)
    assert r1["score"] == 1.0 and r1["cite_precision"] == 1.0 and r1["recall_at_k"] == 1.0
    assert "[pump_A.pdf p.2]" in vlm.calls[0]["prompt"] and vlm.calls[0]["n_images"] == 0

    r2 = score_row(run_b2(q, store, vlm, n_pages=8), q, "rule", vlm)
    assert r2["score"] == 1.0 and r2["recall_at_k"] is None and vlm.calls[1]["n_images"] == 4

    corpus_q = Question("q2", "manuals", "?", "x", doc_ids=[])
    assert run_b2(corpus_q, store, vlm)["skipped"]
