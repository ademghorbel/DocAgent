"""DocAgent graph with a scripted fake VLM (no GPU, no downloads)."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("langgraph")

from docagent.agent import DocAgent, rrf, run_agent  # noqa: E402
from docagent.data import UNANSWERABLE, Question  # noqa: E402
from docagent.retrieve import TextIndex  # noqa: E402
from docagent.visual import VisualIndex  # noqa: E402
from docagent.vlm import GenResult  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from run_eval import score_row  # noqa: E402
from test_cpu_parts import HashEmbedder, store  # noqa: E402,F401


class ScriptedVLM:
    """Replies by step: reading prompts get a note (or NONE), the answer and verify prompts get fixed replies."""

    def __init__(self, notes: dict, answer="Answer: x\nPages: none", verify=("SUPPORTED",)):
        self.notes, self.answer, self.verify = notes, answer, list(verify)
        self.calls = []

    def generate(self, prompt, images=None, system=None, max_new_tokens=256):
        labels = [lab for lab, _ in images or []]
        if "Copy out every fact" in prompt:
            kind, text = "read", self.notes.get(labels[0], "NONE")
        elif "Do the notes fully support" in prompt:
            kind, text = "verify", self.verify.pop(0) if len(self.verify) > 1 else self.verify[0]
        else:
            kind, text = "answer", self.answer
        self.calls.append((kind, labels))
        return GenResult(text, 1.0, 50, 2.0)


Q = Question("q1", "manuals", "What does error E-47 mean?", "discharge pressure too high", "Str",
             doc_ids=["pump_A.pdf"], evidence=[{"doc_id": "pump_A.pdf", "page": 2}], tags=["table"])


def visual_index(store, best_index):
    keys = [p.key for p in store.pages]
    return VisualIndex(keys, np.eye(len(keys), dtype=np.float32), np.arange(len(keys)),
                       {"q1": np.eye(len(keys), dtype=np.float32)[[best_index]]})


def test_rrf_rewards_pages_both_retrievers_like():
    fused = rrf([[("a", 1), ("a", 2), ("a", 3)], [("a", 2), ("a", 9), ("a", 1)]])
    assert fused[0] == ("a", 2) and set(fused) == {("a", 1), ("a", 2), ("a", 3), ("a", 9)}


def test_agent_reads_pages_one_by_one_and_answers(store):  # noqa: F811
    vlm = ScriptedVLM({"[pump_A.pdf p.2]": "E-47: discharge pressure too high"},
                      answer="Answer: Discharge pressure too high\nPages: [pump_A.pdf p.2]")
    agent = DocAgent(store, vlm, TextIndex.build(store, HashEmbedder()), visual_index(store, 1), pages_per_round=3)
    row = score_row(run_agent(Q, agent), Q, "rule", vlm)
    reads = [c for c in vlm.calls if c[0] == "read"]
    assert len(reads) == 3 and all(len(labels) == 1 for _, labels in reads)   # one page per call
    assert row["retrieved"][0] == ["pump_A.pdf", 2]                          # both retrievers agree on p.2
    assert row["score"] == 1.0 and row["cite_precision"] == 1.0 and row["rounds"] == 1
    answer_call = next(c for c in vlm.calls if c[0] == "answer")
    assert answer_call[1] == ["[pump_A.pdf p.2]"]   # only pages with useful notes are shown again
    assert row["system"] == "DA_agent" and any("verify: supported" in t for t in row["trace"])


def test_agent_says_not_answerable_when_no_page_helps(store):  # noqa: F811
    vlm = ScriptedVLM({})   # every page: NONE
    agent = DocAgent(store, vlm, TextIndex.build(store, HashEmbedder()), None, pages_per_round=2)
    row = run_agent(Q, agent)
    assert row["prediction"] == UNANSWERABLE and row["verdict"] == "no_evidence"
    assert row["rounds"] == 2 and row["pages_used"] == 3   # looked further once; blank p.3 has no text to find
    assert not any(c[0] == "answer" for c in vlm.calls)


def test_verifier_triggers_a_second_round(store):  # noqa: F811
    vlm = ScriptedVLM({"[pump_A.pdf p.1]": "pump P-12", "[pump_A.pdf p.2]": "E-47: discharge pressure too high"},
                      answer="Answer: Discharge pressure too high\nPages: [pump_A.pdf p.2]",
                      verify=("MISSING: error E-47 troubleshooting table", "SUPPORTED"))
    agent = DocAgent(store, vlm, TextIndex.build(store, HashEmbedder()), visual_index(store, 0), pages_per_round=1)
    row = run_agent(Q, agent)
    assert row["rounds"] == 2 and row["pages_used"] == 2
    assert [c[0] for c in vlm.calls] == ["read", "answer", "verify", "read", "answer", "verify"]
    assert any("search 'error E-47 troubleshooting table'" in t for t in row["trace"])
    assert row["prediction"] == "Discharge pressure too high"
