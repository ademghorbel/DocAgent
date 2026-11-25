"""DocAgent: a small LangGraph agent built on what the baselines got wrong.

What the B1/B2/B3 run showed (results/baselines/analysis.md):
- Retrieval is mostly fine, reading is not. ColQwen puts a gold page in the top 5 for 84% of
  answerable questions, yet B3 answers only 54% of those right.
- Text and images are complementary. B1 or B3 is right on 69% of questions, each alone on 50-54%.
- B3 rarely says "Not answerable" when it should (11 of 19), B1 refuses too often (24 of 132).

So the agent:
  retrieve  fuse ColQwen and text rankings (reciprocal rank fusion), take the next unread pages
  read      look at ONE page at a time, image plus its text layer, and copy out the facts that help
  answer    answer from those notes with the top pages shown together (pages with notes first),
            so facts spread over several pages can be compared side by side
  verify    check the answer against the notes; if something is missing, write a follow-up search
            and go round once more. If no page had anything relevant, answer "Not answerable".

The same VLM does every step, so a gain over B3 comes from the procedure, not a bigger model.

After the first full run (DocAgent 54.3%, same as B3, but 40.3 vs 50.6 on multi-page questions and
page recall@5 0.59 vs 0.64), two changes:
- the answer step shows up to CFG.direct_vlm_pages page images at once, like B3, instead of two.
  Notes are per page, so anything that needs two pages side by side was lost;
- ColQwen counts double in the fusion (VISUAL_WEIGHT), since it is the stronger retriever
  (84% hit rate vs 72% for text) and equal weights pulled recall below B3's.

v2 on 184 questions: 61.4% (B3 59.2), but still behind B3 on multi-page questions (43.8 vs 51.2) and on
ViDoRe (61.7 vs 71.7). v3 targets those two:
- a cross-encoder reranker re-scores the top RERANK_DEPTH fused pages from their text (question and page
  read together), and its order is fused with the original one, so a page only the image search found
  is not lost;
- neighbouring pages: when a page has useful facts, the page before and after it are read too (up to
  NEIGHBOURS extra pages per round). Tables and procedures often continue on the next page.
Each change has a switch (DOCAGENT_RERANK="", DOCAGENT_NEIGHBOURS=0, DOCAGENT_VERIFY=0,
DOCAGENT_VISUAL_WEIGHT=0) so an ablation run can measure what each part brings.
"""
from __future__ import annotations

import re
from typing import TypedDict

from PIL import Image

from .config import CFG, _env
from .data import UNANSWERABLE, PageStore, Question
from .prompts import SYSTEM, answer_prompt, page_label, parse_reply

PAGES_PER_ROUND = 5        # pages read closely per round
MAX_ROUNDS = 2             # round 2 only happens when the verifier asks for more
PAGE_TEXT_CHARS = 2500     # text layer passed alongside each page image
ANSWER_IMAGES = _env("DOCAGENT_ANSWER_IMAGES", CFG.direct_vlm_pages, int)   # page images at the answer step (B3's budget)
VISUAL_WEIGHT = _env("DOCAGENT_VISUAL_WEIGHT", 2.0, float)                  # ColQwen's weight in the fusion, text = 1
RRF_K = 60                 # standard reciprocal-rank-fusion constant
RERANK_DEPTH = 20          # fused pages re-scored by the cross-encoder
NEIGHBOURS = _env("DOCAGENT_NEIGHBOURS", 2, int)   # extra adjacent pages read per round (0 = off)
VERIFY = _env("DOCAGENT_VERIFY", "1") == "1"       # 0 = answer once, no verifier, no second round
# Name written to predictions.jsonl. A rerun skips (system, question) pairs that already have an
# answer, so a changed agent needs a new name (v1 rows stay "DA_agent", v2 "DA_agent_v2").
# Ablation runs set DOCAGENT_SYSTEM_NAME, e.g. "DA_v3_no_rerank".
SYSTEM_NAME = _env("DOCAGENT_SYSTEM_NAME", "DA_agent_v3")

READ_PROMPT = """Question: {question}

Above is one page{text_note}. Copy out every fact on this page that helps answer the question:
exact numbers with units, names, table rows with their column headers, labels and counts from
figures. A fact that answers only part of the question counts (one device of a comparison, one
possible cause of a symptom: fault codes, their causes and the settings involved). Keep it short.
Do not answer the question yet.
If nothing on this page helps, reply exactly: NONE"""

# The reader sees one page; the shared SYSTEM prompt made it reply "Not answerable" for a page that
# held half of the answer, and those replies were kept as notes.
READER_SYSTEM = ("You read one page of a technical document and copy out the facts on it that are "
                 "relevant to a question. Pages are labelled like [doc.pdf p.12].")

VERIFY_PROMPT = """Question: {question}
Proposed answer: {answer}

Notes taken from the pages:
{notes}

Do the notes fully support the proposed answer?
Reply SUPPORTED if they do.
Otherwise reply MISSING: <a short search query for the information that is still missing>"""

# Demo only: a question that names no device can match several manuals that disagree
# ("what battery does the terminal use": CR2032 in the ATV320, CR1025 in the S7-1200).
SEVERAL_DEVICES = ("The question names no device and the notes come from manuals of different devices "
                   "({devices}). If they give different answers, give each device's answer separately, "
                   "naming the device; do not pick one.")


class AgentState(TypedDict, total=False):
    question: Question
    ranking: list[tuple[str, int]]     # fused page ranking, best first
    first_pages: list[tuple[str, int]] # pages read before the ranking (e.g. a fault-catalog hit)
    batch: list[tuple[str, int]]       # pages to read in this round
    read: list[tuple[str, int]]        # pages already read, in reading order
    notes: dict                        # (doc_id, page) -> note text (only pages with something useful)
    follow_up: str                     # search query written by the verifier
    round: int
    answer: str
    cited: list[tuple[str, int]]
    verdict: str                       # supported | missing | no_evidence
    raw: str
    trace: list[str]                   # one line per step, for the demo and error analysis
    latency_s: float
    input_tokens: int
    peak_gpu_gb: float


def rrf(rankings: list[list[tuple[str, int]]], weights: list[float] | None = None,
        k: int = RRF_K) -> list[tuple[str, int]]:
    """Reciprocal rank fusion: a page ranked high by any retriever ends up near the top.
    weights (one per ranking, default 1) make one retriever count more than another."""
    score: dict[tuple[str, int], float] = {}
    for ranking, w in zip(rankings, weights or [1.0] * len(rankings)):
        for rank, key in enumerate(ranking):
            score[key] = score.get(key, 0.0) + w / (k + rank + 1)
    return sorted(score, key=lambda key: -score[key])


_NOTHING = re.compile(r"\W*(none|not answerable|n/?a)\W*|\W*(the|this) page does not (contain|provide|mention)\b.*",
                      re.I | re.S)


def _is_none(note: str) -> bool:
    # ponytail: a hedge that still quotes numbers ("does not give 100 °C, but ±0.15 °C to 70 °C") is a fact
    note = note.strip()
    return not note or (_NOTHING.fullmatch(note) is not None and not re.search(r"(?<!\w)\d", note))   # not model names like TMP117


class DocAgent:
    def __init__(self, store: PageStore, vlm, text_index=None, visual_index=None,
                 pages_per_round: int = PAGES_PER_ROUND, max_rounds: int = MAX_ROUNDS, depth: int = 20,
                 answer_images: int = ANSWER_IMAGES, visual_weight: float = VISUAL_WEIGHT,
                 compare_devices: bool = False, reranker=None, neighbours: int = NEIGHBOURS,
                 verify: bool = VERIFY):
        if text_index is None and visual_index is None:
            raise ValueError("DocAgent needs at least one retriever")
        self.store, self.vlm = store, vlm
        self.text_index, self.visual_index = text_index, visual_index
        self.pages_per_round, self.max_rounds, self.depth = pages_per_round, max_rounds, depth
        self.answer_images, self.visual_weight = answer_images, visual_weight
        self.compare_devices = compare_devices
        self.reranker, self.neighbours, self.verify_on = reranker, neighbours, verify
        self.graph = self._build_graph()

    # ---------- helpers ----------
    def _gen(self, state: AgentState, prompt: str, images=None, max_new_tokens: int = CFG.max_new_tokens,
             system: str = SYSTEM) -> str:
        g = self.vlm.generate(prompt, images=images, system=system, max_new_tokens=max_new_tokens)
        state["latency_s"] = state.get("latency_s", 0.0) + g.latency_s
        state["input_tokens"] = state.get("input_tokens", 0) + g.input_tokens
        state["peak_gpu_gb"] = max(state.get("peak_gpu_gb", 0.0), g.peak_gpu_gb)
        return g.text

    def _text_pages(self, query: str, doc_ids) -> list[tuple[str, int]]:
        if self.text_index is None:
            return []
        from .retrieve import unique_pages
        return unique_pages(self.text_index.search(query, k=self.depth * 3, doc_ids=doc_ids or None))[: self.depth]

    def _rerank(self, question: str, ranking: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """Fuse the ranking with the cross-encoder's order of its top pages (pages without text keep
        their place through the original ranking)."""
        top = [k for k in ranking[:RERANK_DEPTH] if (p := self.store.get(*k)) and (p.text or "").strip()]
        if self.reranker is None or len(top) < 2:
            return ranking
        scores = self.reranker.scores(question, [self.store.get(*k).text[:PAGE_TEXT_CHARS] for k in top])
        reranked = [k for _, k in sorted(zip(scores, top), key=lambda x: -x[0])]
        return rrf([ranking, reranked])

    def _neighbours(self, state: AgentState, keys) -> list[tuple[str, int]]:
        """Unread pages just before / after pages that gave facts, nearest first."""
        out = []
        for doc, page in keys:
            for d in (1, -1):
                k = (doc, page + d)
                if k not in state["read"] and k not in out and k not in keys and self.store.get(*k):
                    out.append(k)
        return out[: self.neighbours]

    def _image(self, key):
        page = self.store.get(*key)
        return Image.open(self.store.image_path(page)) if page else None

    # ---------- nodes ----------
    def retrieve(self, state: AgentState) -> AgentState:
        q = state["question"]
        if state.get("round", 0) == 0:
            rankings, weights = [], []
            if self.visual_index is not None and self.visual_weight > 0 and q.qid in self.visual_index.queries:
                rankings.append([key for key, _ in self.visual_index.search(q.qid, k=self.depth, doc_ids=q.doc_ids)])
                weights.append(self.visual_weight)
            rankings.append(self._text_pages(q.question, q.doc_ids))
            weights.append(1.0)
            first = [k for k in state.get("first_pages") or [] if self.store.get(*k)]
            fused = self._rerank(q.question, rrf(rankings, weights))
            state["ranking"] = first + [k for k in fused if k not in first]
            state.update(read=[], notes={}, trace=[])
        else:
            # Round 2: pages found by the verifier's follow-up query come first, then the rest of the ranking.
            extra = self._text_pages(state.get("follow_up") or q.question, q.doc_ids)
            state["ranking"] = rrf([extra, state["ranking"]])
        state["round"] = state.get("round", 0) + 1
        todo = [k for k in state["ranking"] if k not in state["read"] and self.store.get(*k)]
        state["batch"] = todo[: self.pages_per_round]
        state["trace"].append(f"round {state['round']}: reading {[page_label(*k) for k in state['batch']]}")
        return state

    def read(self, state: AgentState) -> AgentState:
        batch = state.pop("batch", [])
        self._read_pages(state, batch)
        if self.neighbours:
            extra = self._neighbours(state, [k for k in batch if k in state["notes"]])
            if extra:
                state["trace"].append(f"next to pages with facts: reading {[page_label(*k) for k in extra]}")
                self._read_pages(state, extra)
        return state

    def _read_pages(self, state: AgentState, keys) -> None:
        q = state["question"]
        for key in keys:
            page = self.store.get(*key)
            text = (page.text or "").strip()[:PAGE_TEXT_CHARS]
            prompt = (f"Text layer of this page:\n{text}\n\n" if text else "") + READ_PROMPT.format(
                question=q.question, text_note=" and its text layer" if text else "")
            note = self._gen(state, prompt, images=[(page_label(*key), self._image(key))], max_new_tokens=160,
                             system=READER_SYSTEM)
            state["read"].append(key)
            if not _is_none(note):
                state["notes"][key] = note.strip()
            state["trace"].append(f"read {page_label(*key)}: {'NONE' if _is_none(note) else note.strip()[:80]}")

    def answer(self, state: AgentState) -> AgentState:
        q = state["question"]
        if not state["notes"]:
            state.update(answer=UNANSWERABLE, cited=[], raw="", verdict="no_evidence")
            state["trace"].append("no page had relevant facts")
            return state
        notes = "\n\n".join(f"{page_label(*k)}\n{v}" for k, v in state["notes"].items())
        # Pages with notes first, then the best-ranked other pages the reader saw, all in one prompt.
        # The extra pages cover a reader that wrongly said NONE, and let the model compare pages.
        best = ([k for k in state["read"] if k in state["notes"]]
                + [k for k in state["read"] if k not in state["notes"]])[: self.answer_images]
        context = f"Notes taken while reading the pages:\n{notes}"
        docs = list(dict.fromkeys(k[0] for k in state["notes"]))
        if self.compare_devices and not q.doc_ids and len(docs) > 1:
            from .catalog import device_name
            context += "\n\n" + SEVERAL_DEVICES.format(devices=", ".join(device_name(d) for d in docs))
        prompt = answer_prompt(q.question, q.answer_format, context)
        while True:   # on out-of-memory, halve the images and retry (4 -> 2 -> 1), as the baselines do
            try:
                raw = self._gen(state, prompt, images=[(page_label(*k), self._image(k)) for k in best])
                break
            except Exception as e:  # noqa: BLE001
                if "out of memory" not in str(e).lower() or len(best) <= 1:
                    raise
                self.vlm.free_memory()
                best = best[: max(1, len(best) // 2)]
                state["trace"].append(f"out of GPU memory, answering with {len(best)} page image(s)")
        ans, cited = parse_reply(raw)
        state.update(answer=ans, cited=cited, raw=raw)
        state["trace"].append(f"answer: {ans[:80]}")
        return state

    def verify(self, state: AgentState) -> AgentState:
        if not self.verify_on:          # ablation: one pass, no check, no second round
            state["verdict"] = state.get("verdict") or "unverified"
            return state
        if state.get("verdict") == "no_evidence" and state["round"] < self.max_rounds:
            state["follow_up"] = state["question"].question
            state["verdict"] = "missing"
            return state
        if state.get("verdict") == "no_evidence":
            return state
        notes = "\n\n".join(f"{page_label(*k)}\n{v}" for k, v in state["notes"].items())
        reply = self._gen(state, VERIFY_PROMPT.format(question=state["question"].question,
                                                      answer=state["answer"], notes=notes), max_new_tokens=40)
        m = re.match(r"\W*missing\s*:?\s*(.*)", reply, re.I | re.S)
        if m:
            lines = m.group(1).strip().splitlines()
            state.update(verdict="missing", follow_up=lines[0][:200] if lines else "")
        else:
            state["verdict"] = "supported"
        state["trace"].append(f"verify: {state['verdict']}" + (f" -> search '{state['follow_up']}'"
                                                                if state["verdict"] == "missing" else ""))
        return state

    def _route(self, state: AgentState) -> str:
        if state["verdict"] == "missing" and state["round"] < self.max_rounds:
            state["verdict"] = ""   # reset for the next round
            return "retrieve"
        return "end"

    def _build_graph(self):
        from langgraph.graph import END, StateGraph
        g = StateGraph(AgentState)
        for name in ("retrieve", "read", "answer", "verify"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "read")
        g.add_edge("read", "answer")
        g.add_edge("answer", "verify")
        g.add_conditional_edges("verify", self._route, {"retrieve": "retrieve", "end": END})
        return g.compile()

    # ---------- public ----------
    def run(self, q: Question, first_pages: list[tuple[str, int]] | None = None) -> AgentState:
        return self.graph.invoke({"question": q, "round": 0, "first_pages": list(first_pages or [])},
                                 {"recursion_limit": 4 * self.max_rounds + 4})


def run_agent(q: Question, agent: DocAgent) -> dict:
    """Same record shape as the baselines, plus the agent's trace."""
    s = agent.run(q)
    # Pages for recall@k: those the agent read, in order (round-1 pages are the top of the fused ranking).
    return {
        "system": SYSTEM_NAME,
        "qid": q.qid,
        "prediction": s.get("answer", ""),
        "raw": s.get("raw", ""),
        "cited": [list(c) for c in s.get("cited", [])],
        "retrieved": [list(k) for k in s.get("read", [])],
        "latency_s": round(s.get("latency_s", 0.0), 2),
        "input_tokens": s.get("input_tokens", 0),
        "peak_gpu_gb": s.get("peak_gpu_gb", 0.0),
        "skipped": "",
        "pages_used": len(s.get("read", [])),
        "rounds": s.get("round", 0),
        "verdict": s.get("verdict", ""),
        "trace": s.get("trace", []),
    }
