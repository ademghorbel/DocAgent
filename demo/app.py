"""DocAgent demo: a simulated machine alert goes in, a sourced diagnosis comes out.

A technician (or the plant's monitoring system) reports a fault code on the Schneider ATV320
drive. DocAgent searches the drive's manual, reads the pages, answers with page citations and
shows the cited pages next to the answer.

Needs the manuals page store and its ColQwen index (run the Kaggle notebook up to the index
cell first). Run:  python demo/app.py --share
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from docagent.agent import DocAgent  # noqa: E402
from docagent.config import CFG  # noqa: E402
from docagent.data import PageStore, Question  # noqa: E402
from docagent.prompts import page_label  # noqa: E402

COLLECTION = "manuals"
DOC = "schneider_atv320_programming.pdf"

# Simulated alerts: what the drive would send. The descriptions are only the alarm text shown
# on the drive's display; the diagnosis itself comes from the manual.
ALERTS = {
    "OCF  overcurrent": "OCF",
    "OHF  drive overheat": "OHF",
    "OLF  motor overload": "OLF",
    "CFF  incorrect configuration": "CFF",
    "USF  supply undervoltage": "USF",
    "PHF  input phase loss": "PHF",
    "SCF1 motor short circuit": "SCF1",
}

QUESTION = ("Machine alert: the ATV320 drive on line 2 stopped and shows the fault code {code}. "
            "What does this fault mean, what are the probable causes, and what should the "
            "technician check or do to clear it?")


class Demo:
    def __init__(self):
        from docagent.retrieve import BGEEmbedder, TextIndex
        from docagent.visual import ColQwen, VisualIndex
        from docagent.vlm import VLM

        self.store = PageStore(CFG.pages_dir, COLLECTION)
        if not len(self.store):
            raise SystemExit(f"No pages in {CFG.pages_dir / COLLECTION}: prepare the manuals first.")
        print(f"{len(self.store)} manual pages", flush=True)
        text_index = TextIndex.for_collection(self.store, BGEEmbedder())
        try:
            self.vindex = VisualIndex.load(COLLECTION)
            self.colqwen = ColQwen()   # embeds the alert text, page vectors are precomputed
        except Exception as e:         # still works on text retrieval alone
            print("No ColQwen index, text retrieval only:", e, flush=True)
            self.vindex, self.colqwen = None, None
        self.agent = DocAgent(self.store, VLM(), text_index, self.vindex)
        self.n = 0

    def ask(self, text: str):
        self.n += 1
        qid = f"demo-{self.n}"
        if self.vindex is not None:
            self.vindex.queries[qid] = self.colqwen.embed_queries([text])[0].astype("float32")
        q = Question(qid=qid, collection=COLLECTION, question=text, answer="",
                     answer_format="Open", doc_ids=[DOC])
        s = self.agent.run(q)
        if self.vindex is not None:
            self.vindex.queries.pop(qid, None)

        cited = [tuple(c) for c in s.get("cited", [])]
        answer = s.get("answer", "")
        sources = ", ".join(page_label(*c) for c in cited) or "none"
        report = (f"### Diagnosis\n\n{answer}\n\n**Sources:** {sources}\n\n"
                  f"*{s.get('latency_s', 0):.0f} s, {len(s.get('read', []))} pages read, "
                  f"{s.get('round', 0) + 1} round(s), verifier: {s.get('verdict', '')}*")
        pages = []
        for c in cited or s.get("read", [])[:2]:
            page = self.store.get(*c)
            if page:
                pages.append((Image.open(self.store.image_path(page)), page_label(*c)))
        return report, pages, "\n".join(s.get("trace", []))


def build_ui(demo: Demo):
    import gradio as gr

    def from_alert(alert):
        return QUESTION.format(code=ALERTS[alert])

    def run(text):
        return demo.ask(text.strip())

    with gr.Blocks(title="DocAgent") as ui:
        gr.Markdown("# DocAgent\nA drive raises an alert; the agent reads the ATV320 manual and "
                    "answers with the pages it used. Pick an alert or type your own question.")
        with gr.Row():
            with gr.Column(scale=2):
                alert = gr.Dropdown(list(ALERTS), value=list(ALERTS)[0], label="Simulated alert")
                text = gr.Textbox(value=from_alert(list(ALERTS)[0]), lines=4, label="Question sent to DocAgent")
                go = gr.Button("Diagnose", variant="primary")
                trace = gr.Textbox(label="Agent trace (pages read, notes, verifier)", lines=12)
            with gr.Column(scale=3):
                report = gr.Markdown()
                pages = gr.Gallery(label="Cited pages", columns=2, height=640)
        alert.change(from_alert, alert, text)
        go.click(run, text, [report, pages, trace])
    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public gradio.live link (needed on Kaggle)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_ui(Demo()).queue().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
