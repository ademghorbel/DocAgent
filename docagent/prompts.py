"""Prompts shared by all systems, and parsing of the model's reply into answer + cited pages."""
from __future__ import annotations

import re

from .data import UNANSWERABLE

SYSTEM = (
    "You answer questions about technical documents using only the pages you are given. "
    "Pages are labelled like [doc.pdf p.12]. "
    f"If the pages do not contain the answer, the answer is exactly: {UNANSWERABLE}."
)

FORMAT_HINT = {
    "Int": "Give a single integer.",
    "Float": "Give a single number (no units unless asked).",
    "List": "Give a short comma-separated list.",
    "Str": "Give a short answer: a value, a name or a short phrase.",
    "None": "Give a short answer.",
    "Open": "Answer in 1 to 4 sentences.",
}

ANSWER_TEMPLATE = """Question: {question}

{hint}
Reply in exactly this format:
Answer: <your answer>
Pages: <labels of the pages you used, e.g. [doc.pdf p.12], or none>"""


def answer_prompt(question: str, answer_format: str = "Str", context: str = "") -> str:
    hint = FORMAT_HINT.get(answer_format, FORMAT_HINT["Str"])
    body = ANSWER_TEMPLATE.format(question=question, hint=hint)
    return f"{context}\n\n{body}" if context else body


def page_label(doc_id: str, page: int) -> str:
    return f"[{doc_id} p.{page}]"


_LABEL = re.compile(r"\[([^\[\]]+?)\s+p\.?\s*(\d+)\]")


def parse_reply(text: str) -> tuple[str, list[tuple[str, int]]]:
    """Split 'Answer: ... Pages: ...' into (answer, [(doc_id, page), ...]). Tolerant of sloppy output."""
    answer_m = re.search(r"answer\s*:\s*(.*?)(?:\n\s*pages?\s*:|$)", text, flags=re.I | re.S)
    if answer_m:
        answer = answer_m.group(1).strip()
        # "<answer text> Answer: [doc.pdf p.3]": the model put its answer before the label
        before = text[: answer_m.start()].strip()
        if before and not _LABEL.sub("", answer).strip(" ,;"):
            answer = before
    else:
        # No "Answer:" label: everything before the "Pages:" line is the answer
        answer = re.split(r"\n\s*pages?\s*:", text, maxsplit=1, flags=re.I)[0].strip()
    pages_m = re.search(r"pages?\s*:\s*(.*)$", text, flags=re.I | re.S)
    cited_from = pages_m.group(1) if pages_m else text[answer_m.start(1):] if answer_m else ""
    cited = [(d.strip(), int(p)) for d, p in _LABEL.findall(cited_from)]
    return answer, list(dict.fromkeys(cited))
