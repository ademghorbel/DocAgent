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


class OOMOnceVLM(FakeVLM):
    """Runs out of memory whenever more than 2 images are passed."""
    def generate(self, prompt, images=None, system=None, max_new_tokens=256):
        if len(images or []) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 7.27 GiB")
        return super().generate(prompt, images, system, max_new_tokens)

    def free_memory(self):
        pass


def test_b2_backs_off_on_oom(store):  # noqa: F811
    q = Question("q1", "manuals", "?", "x", doc_ids=["pump_A.pdf"])
    row = run_b2(q, store, OOMOnceVLM("Answer: x\nPages: none"), n_pages=8)
    assert row["pages_used"] == 2 and row["prediction"] == "x"


def test_select_pages_gives_k_distinct_pages():
    from docagent.baselines import select_pages
    from docagent.retrieve import Hit
    hits = [Hit("a", 1, .9, "x"), Hit("a", 1, .8, "y"), Hit("a", 1, .7, "z"), Hit("a", 2, .6, "w"),
            Hit("b", 5, .5, "v"), Hit("a", 2, .4, "u"), Hit("c", 9, .3, "t")]
    kept, pages = select_pages(hits, k=3)
    assert pages == [("a", 1), ("a", 2), ("b", 5)]
    assert [h.text for h in kept] == ["x", "y", "w", "v", "u"]   # max 2 chunks per page, no 4th page


def test_maxsim_picks_the_page_with_matching_tokens():
    import numpy as np
    from docagent.visual import maxsim_scores
    q = np.array([[1, 0], [0, 1]], dtype=np.float16)
    # page 0: two tokens matching nothing well; page 1: one token per query token
    tokens = np.array([[0.1, 0.1], [0.2, 0.0], [1, 0], [0, 1], [0.5, 0.5]], dtype=np.float16)
    scores = maxsim_scores(q, tokens, np.array([0, 2]))
    assert scores.argmax() == 1 and abs(scores[1] - 2.0) < 1e-3


def test_b3_end_to_end(store):  # noqa: F811
    import numpy as np
    from docagent.baselines import run_b3
    from docagent.visual import VisualIndex
    keys = [p.key for p in store.pages]
    tokens = np.eye(len(keys), dtype=np.float32)            # one token per page, page i = axis i
    q2 = np.eye(len(keys), dtype=np.float32)[[1]]           # the query matches page 2 (index 1)
    vindex = VisualIndex(keys, tokens, np.arange(len(keys)), {"q1": q2})
    q = Question("q1", "manuals", "What does error E-47 mean?", "discharge pressure too high", "Str",
                 doc_ids=["pump_A.pdf"], evidence=[{"doc_id": "pump_A.pdf", "page": 2}], tags=["table"])
    vlm = FakeVLM("Answer: Discharge pressure too high\nPages: [pump_A.pdf p.2]")
    row = score_row(run_b3(q, store, vindex, vlm, k=3, n_pages=2), q, "rule", vlm)
    assert row["system"] == "B3_colqwen_rag" and row["retrieved"][0] == ["pump_A.pdf", 2]
    assert row["score"] == 1.0 and row["recall_at_k"] == 1.0 and vlm.calls[0]["n_images"] == 2
