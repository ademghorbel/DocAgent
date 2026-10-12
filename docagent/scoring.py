"""Scoring: answer correctness, page recall@k, citation correctness, and the results table.

Short answers (MMLongBench-Doc, most own questions) are scored by rules, no LLM needed.
Open answers (ViDoRe V3) need a judge; `llm_judge` uses a rubric and returns 0 or 1.
"""
from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Callable

from .data import TAGS, UNANSWERABLE, Question

_UNANSWERABLE_HINTS = ("not answerable", "cannot be answered", "not mentioned", "no information",
                       "not provided", "does not contain", "not found", "not specified", "unanswerable")


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\"'`*]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .;:")


def says_unanswerable(pred: str) -> bool:
    p = normalize(pred)
    return any(h in p for h in _UNANSWERABLE_HINTS)


def _numbers(s: str) -> list[float]:
    s = s.replace(",", "") if re.search(r"\d,\d{3}\b", s) else s.replace(",", ".")
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]


def _num_match(gold: str, pred: str, rel_tol: float = 0.01) -> bool:
    g, p = _numbers(gold), _numbers(pred)
    if not g or not p:
        return False
    target = g[0]
    is_pct = "%" in gold
    for x in p:
        candidates = [x, x * 100, x / 100] if is_pct else [x]
        if any(abs(c - target) <= max(rel_tol * abs(target), 1e-6) for c in candidates):
            return True
    return False


def _str_match(gold: str, pred: str) -> bool:
    g, p = normalize(gold), normalize(pred)
    if not g:
        return False
    if g == p or (len(g) >= 3 and g in p and len(p) <= 4 * len(g) + 20):
        return True
    if _numbers(g) and _num_match(g, p):
        return True
    return SequenceMatcher(None, g, p).ratio() >= 0.85


def _list_items(s: str) -> list[str]:
    s = s.strip().strip("[]")
    return [normalize(x) for x in re.split(r",|;|\n| and ", s) if normalize(x)]


def rule_score(q: Question, pred: str) -> float | None:
    """1.0 correct, 0.0 wrong, None when rules can't decide (open answers)."""
    gold_unans = normalize(q.answer) == normalize(UNANSWERABLE)
    pred_unans = says_unanswerable(pred)
    if gold_unans or pred_unans:
        return float(gold_unans and pred_unans)
    fmt = q.answer_format
    if fmt == "Open":
        return None
    if fmt in ("Int", "Float"):
        return float(_num_match(q.answer, pred))
    if fmt == "List":
        gold_items = _list_items(q.answer)
        pred_text = normalize(pred)
        found = sum(1 for g in gold_items if g in pred_text or any(_str_match(g, p) for p in _list_items(pred)))
        return float(bool(gold_items) and found == len(gold_items))
    return float(_str_match(q.answer, pred))


JUDGE_PROMPT = """You are grading an answer to a question about technical documents.

Question: {question}
Reference answer: {gold}
Candidate answer: {pred}

The candidate is CORRECT if it states the same key facts as the reference (values, names, steps),
even with different wording or extra detail. It is WRONG if a key fact is missing, different, or
if it refuses when the reference has an answer.
Reply with one word: CORRECT or WRONG."""


def llm_judge(q: Question, pred: str, generate: Callable[[str], str]) -> float:
    """generate: prompt -> reply text. Hand-check ~20 judgments before trusting it (see README)."""
    reply = generate(JUDGE_PROMPT.format(question=q.question, gold=q.answer, pred=pred))
    return 1.0 if re.search(r"\bcorrect\b", reply, re.I) and not re.search(r"\bwrong\b", reply, re.I) else 0.0


# ---------- retrieval and citation ----------

def recall_at_k(retrieved: list[tuple[str, int]], evidence: set[tuple[str, int]], k: int) -> float | None:
    if not evidence:
        return None
    top = {tuple(r) for r in retrieved[:k]}
    return len(top & evidence) / len(evidence)


def citation_scores(cited: list[tuple[str, int]], evidence: set[tuple[str, int]]) -> tuple[float | None, float | None]:
    """(precision, recall) of cited pages vs gold evidence pages."""
    if not evidence:
        return None, None
    cited_set = {tuple(c) for c in cited}
    if not cited_set:
        return 0.0, 0.0
    hit = len(cited_set & evidence)
    return hit / len(cited_set), hit / len(evidence)


# ---------- table ----------

def summary_table(rows: list[dict], questions: dict[str, Question], k: int = 5) -> str:
    """Markdown table: accuracy per system x tag, plus recall@k, citation precision and latency."""
    acc = defaultdict(lambda: defaultdict(list))
    extra = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("skipped") or r.get("score") is None:
            continue
        q = questions[r["qid"]]
        s = r["system"]
        acc[s]["all"].append(r["score"])
        for t in q.tags:
            acc[s][t].append(r["score"])
        for key in ("recall_at_k", "cite_precision", "latency_s", "peak_gpu_gb"):
            if r.get(key) is not None:
                extra[s][key].append(r[key])

    def pct(v):
        return f"{100 * sum(v) / len(v):.1f} ({len(v)})" if v else "–"

    def mean(v, fmt="{:.2f}"):
        return fmt.format(sum(v) / len(v)) if v else "–"

    cols = ["all", *TAGS]
    head = "| System | " + " | ".join(cols) + f" | Page recall@{k} | Citation precision | Latency (s) | Peak GPU (GB) |"
    sep = "|" + "---|" * (len(cols) + 5)
    lines = [head, sep]
    for s in sorted(acc):
        e = extra[s]
        lines.append(f"| {s} | " + " | ".join(pct(acc[s][c]) for c in cols) +
                     f" | {mean(e['recall_at_k'], '{:.2f}')} | {mean(e['cite_precision'], '{:.2f}')}"
                     f" | {mean(e['latency_s'], '{:.1f}')} | {mean(e['peak_gpu_gb'], '{:.1f}')} |")
    lines.append("")
    lines.append("Accuracy in %, number of scored questions in brackets. "
                 "A question can carry several tags, so tag columns don't add up to `all`.")
    return "\n".join(lines)
