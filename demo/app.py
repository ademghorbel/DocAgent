"""DocAgent demo: an alert or a question goes in, a sourced answer comes out.

- Exact fault codes (OCF, 2310, ...) are answered instantly from the fault catalog, with the page.
- Anything else (symptoms, settings, comparisons) goes to the agent, which reads the manuals.
- The device named in the question picks the manual ("ACS580 fault 2310" only searches the ABB
  manual); no device named means all manuals.
- A new manual can be uploaded: it is indexed and catalogued on the spot.
- The catalog tab lists every code of every manual. Missing rows are built in the background
  at launch (the VLM reads each manual's fault pages once; saved, so a restart keeps them).

Needs the manuals page store and its ColQwen index (run the Kaggle notebook up to the index
cell first). Run:  python demo/app.py --share
"""
from __future__ import annotations

import argparse
import shutil
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from docagent.agent import DocAgent  # noqa: E402
from docagent.catalog import Catalog, Fault, build, device_name, route  # noqa: E402
from docagent.config import CFG  # noqa: E402
from docagent.data import UNANSWERABLE, PageStore, Question  # noqa: E402
from docagent.prompts import page_label  # noqa: E402

COLLECTION = "manuals"
AUTO, ALL = "auto", "all"
ATV320 = "schneider_atv320_programming.pdf"
# Until the catalog has rows, the alert list falls back to these ATV320 codes.
FALLBACK_ALERTS = [("OCF", "overcurrent"), ("OHF", "drive overheat"), ("OLF", "motor overload"),
                   ("CFF", "incorrect configuration"), ("USF", "supply undervoltage"),
                   ("PHF", "input phase loss"), ("SCF1", "motor short circuit")]
QUESTION = ("Machine alert: the {device} stopped and shows the fault code {code}. What does this fault "
            "mean, what are the probable causes, and what should the technician check or do to clear it?")
EXAMPLES = [
    "The ATV320 trips every time the motor accelerates. What should I check?",
    "ABB ACS580 shows fault 2310. What does it mean and how do I fix it?",
    "How do I set the acceleration ramp time on the ATV320?",
    "Which temperature sensor is more accurate around 100 °C, the TMP117 or the LM35?",
    "What does ISA-18.2 say about alarm flooding?",
]


class LockedVLM:
    """The demo shares one VLM between the user's questions and the background catalog build."""

    def __init__(self, vlm, lock):
        self.vlm, self.lock = vlm, lock

    def generate(self, *a, **kw):
        with self.lock:
            return self.vlm.generate(*a, **kw)

    def free_memory(self):
        self.vlm.free_memory()


class Demo:
    def __init__(self, build_catalog: bool = True):
        from docagent.retrieve import BGEEmbedder, TextIndex
        from docagent.visual import ColQwen, VisualIndex
        from docagent.vlm import VLM

        self.gpu = threading.RLock()
        self.store = PageStore(CFG.pages_dir, COLLECTION)
        if not len(self.store):
            raise SystemExit(f"No pages in {CFG.pages_dir / COLLECTION}: prepare the manuals first.")
        print(f"{len(self.store)} manual pages", flush=True)
        self.text_index = TextIndex.for_collection(self.store, BGEEmbedder())
        try:
            self.vindex = VisualIndex.load(COLLECTION)
            self.colqwen = ColQwen()   # embeds the question (and uploaded pages); page vectors are precomputed
        except Exception as e:         # still works on text retrieval alone
            print("No ColQwen index, text retrieval only:", e, flush=True)
            self.vindex, self.colqwen = None, None
        self.vlm = LockedVLM(VLM(), self.gpu)
        self.agent = DocAgent(self.store, self.vlm, self.text_index, self.vindex)
        self.catalog = Catalog()
        self.status = f"Catalog: {len(self.catalog)} codes from {len(self.catalog.done)} manuals."
        self.n = 0
        if build_catalog and set(self.store.doc_ids()) - self.catalog.done:
            # ATV320 first: the alert list and the demo questions start there
            todo = sorted(set(self.store.doc_ids()) - self.catalog.done, key=lambda d: (d != ATV320, d))
            threading.Thread(target=self._build, args=(todo,), daemon=True).start()

    # ---------- catalog ----------
    def _progress(self, msg: str):
        self.status = f"Building the catalog: {msg}  ({len(self.catalog)} codes so far)"
        print(msg, flush=True)

    def _build(self, docs):
        try:
            build(self.store, self.vlm, self.catalog, docs, progress=self._progress)
            self.status = f"Catalog: {len(self.catalog)} codes from {len(self.catalog.done)} manuals."
        except Exception as e:  # noqa: BLE001  the demo keeps working without the rest of the catalog
            self.status = f"Catalog build stopped: {e}"
            print(self.status, flush=True)

    def docs(self) -> list[tuple[str, str]]:
        return [(device_name(d), d) for d in self.store.doc_ids()]

    def alerts(self) -> list[tuple[str, str]]:
        """(label, 'doc_id|code') for the alert dropdown."""
        rows = sorted(self.catalog.faults, key=lambda x: (x.doc_id != ATV320, x.device, x.code))
        if rows:
            return [(f"{x.device} · {x.code} · {x.name[:50]}", f"{x.doc_id}|{x.code}") for x in rows]
        return [(f"Schneider ATV320 · {c} · {n}", f"{ATV320}|{c}") for c, n in FALLBACK_ALERTS]

    def table(self, text: str = "") -> list[list]:
        return [[x.device, x.code, x.name, x.causes, x.remedy, ", ".join(map(str, x.pages))]
                for x in self.catalog.search(text)]

    # ---------- answering ----------
    def scope(self, text: str, choice: str) -> list[str]:
        if choice == ALL:
            return []
        if choice == AUTO:
            return route(text, self.store.doc_ids())
        return [choice]

    def _pages(self, keys, n=4):
        out = []
        for k in keys[:n]:
            page = self.store.get(*k)
            if page:
                out.append((Image.open(self.store.image_path(page)), page_label(*k)))
        return out

    def _from_catalog(self, hits: list[Fault], where: str):
        parts = []
        for x in hits:
            src = ", ".join(page_label(x.doc_id, p) for p in x.pages)
            parts.append(f"#### {x.device}: fault {x.code}" + (f", {x.name}" if x.name else "")
                         + (f"\n\n**Probable causes:** {x.causes}" if x.causes else "")
                         + (f"\n\n**What to do:** {x.remedy}" if x.remedy else "")
                         + f"\n\n**Source:** {src}")
        report = ("### From the fault catalog (exact code match)\n\n" + "\n\n".join(parts)
                  + f"\n\n*Instant, no model call. Searched: {where}. Untick \"Fault catalog first\" "
                    "to have the agent read the pages instead.*")
        keys = [(x.doc_id, p) for x in hits for p in x.pages[:1]]
        trace = f"routed to: {where}\ncatalog: " + ", ".join(f"{x.device} {x.code}" for x in hits)
        return report, self._pages(keys), trace

    def ask(self, text: str, choice: str = AUTO, use_catalog: bool = True):
        text = text.strip()
        if not text:
            return "Type a question or pick an alert.", [], ""
        doc_ids = self.scope(text, choice)
        where = ", ".join(device_name(d) for d in doc_ids) if doc_ids else "all manuals"
        if use_catalog:
            hits = self.catalog.find_codes(text, doc_ids or None)
            if hits:
                return self._from_catalog(hits[:4], where)

        self.n += 1
        qid = f"demo-{self.n}"
        if self.vindex is not None:
            with self.gpu:
                self.vindex.queries[qid] = self.colqwen.embed_queries([text])[0].astype("float32")
        q = Question(qid=qid, collection=COLLECTION, question=text, answer="",
                     answer_format="Open", doc_ids=doc_ids)
        try:
            s = self.agent.run(q)
        finally:
            if self.vindex is not None:
                self.vindex.queries.pop(qid, None)

        trace = f"routed to: {where}\n" + "\n".join(s.get("trace", []))
        answer = s.get("answer", "")
        if answer.strip().lower().startswith(UNANSWERABLE.lower()):
            loaded = ", ".join(n for n, _ in self.docs())
            return (f"### Not in the loaded manuals\n\nNothing in {where} answers this, so DocAgent "
                    f"does not guess.\n\n**Loaded manuals:** {loaded}\n\nUpload the right manual in the "
                    f"\"Add a manual\" tab and ask again.", [], trace)
        cited = [tuple(c) for c in s.get("cited", [])]
        sources = ", ".join(page_label(*c) for c in cited) or "none"
        report = (f"### Answer\n\n{answer}\n\n**Sources:** {sources}\n\n"
                  f"*Searched: {where}. {s.get('latency_s', 0):.0f} s, {len(s.get('read', []))} pages read, "
                  f"{s.get('round', 0)} round(s), verifier: {s.get('verdict', '')}*")
        return report, self._pages(cited or s.get("read", [])[:2]), trace

    # ---------- new manual ----------
    def add_manual(self, pdf_path: str, progress=print) -> str:
        from docagent.ingest import ingest_pdf
        from docagent.visual import save_shard

        name = Path(pdf_path).name
        if name in self.store.doc_ids():
            return f"{name} is already loaded."
        dest = CFG.pdf_dir / COLLECTION / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf_path, dest)

        progress(0.05, f"Rendering the pages of {name}")
        ingest_pdf(dest, self.store, doc_id=name)
        self.store.save()
        pages = self.store.doc_pages(name)
        progress(0.15, f"{len(pages)} pages: text index")
        self.text_index.add_pages(pages)
        self.text_index.save_for(self.store)

        if self.colqwen is not None:
            vecs = []
            for i in range(0, len(pages), 4):
                with self.gpu:
                    vecs += self.colqwen.embed_images([Image.open(self.store.image_path(p)) for p in pages[i:i + 4]])
                progress(0.15 + 0.45 * min(i + 4, len(pages)) / len(pages),
                         f"Visual index: {min(i + 4, len(pages))}/{len(pages)} pages")
            keys = [p.key for p in pages]
            self.vindex.add(keys, vecs)
            save_shard(COLLECTION, keys, vecs, len(self.store))

        progress(0.6, "Reading the fault pages for the catalog")
        before = len(self.catalog)
        build(self.store, self.vlm, self.catalog, [name], progress=lambda m: progress(0.8, m))
        return (f"**{device_name(name)}** added: {len(pages)} pages, "
                f"{len(self.catalog) - before} fault codes in the catalog. You can ask about it now.")


def build_ui(demo: Demo):
    import gradio as gr

    scope_choices = lambda: [("Auto: the device named in the question", AUTO), ("All manuals", ALL)] + demo.docs()  # noqa: E731

    def from_alert(value):
        if not value:
            return gr.update(), gr.update()
        doc, code = value.split("|", 1)
        return QUESTION.format(device=device_name(doc), code=code), doc

    def run(text, choice, use_catalog):
        return demo.ask(text, choice, use_catalog)

    def upload(file, progress=gr.Progress()):
        if not file:
            return "Pick a PDF first.", gr.update(), gr.update()
        path = file if isinstance(file, str) else file.name
        msg = demo.add_manual(path, progress=lambda f, m: progress(f, desc=m))
        return msg, gr.update(choices=scope_choices()), gr.update(value=demo.table())

    def refresh(filter_text):
        return demo.status, gr.update(choices=demo.alerts()), demo.table(filter_text)

    with gr.Blocks(title="DocAgent") as ui:
        gr.Markdown("# DocAgent\nAsk about any loaded manual, or pick a machine alert. Exact fault codes "
                    "come straight from the fault catalog; everything else is read from the pages by "
                    "the agent. Every answer shows the pages it comes from.")
        status = gr.Markdown(demo.status)
        with gr.Tab("Ask"):
            with gr.Row():
                with gr.Column(scale=2):
                    alert = gr.Dropdown(demo.alerts(), value=None, label="Simulated alert (optional)",
                                        filterable=True)
                    text = gr.Textbox(lines=4, label="Question", placeholder="Ask anything about the manuals")
                    gr.Examples(EXAMPLES, inputs=text)
                    with gr.Row():
                        scope = gr.Dropdown(scope_choices(), value=AUTO, label="Search in")
                        use_catalog = gr.Checkbox(True, label="Fault catalog first")
                    go = gr.Button("Ask", variant="primary")
                    trace = gr.Textbox(label="Trace (routing, pages read, notes, verifier)", lines=10)
                with gr.Column(scale=3):
                    report = gr.Markdown()
                    pages = gr.Gallery(label="Source pages", columns=2, height=640)
        with gr.Tab("Fault catalog"):
            find = gr.Textbox(label="Filter (device, code or any word)", placeholder="e.g. ATV320 overcurrent")
            table = gr.Dataframe(demo.table(), headers=["Device", "Code", "Name", "Probable causes",
                                                        "What to do", "Pages"], wrap=True, interactive=False)
            again = gr.Button("Refresh")
        with gr.Tab("Add a manual"):
            gr.Markdown("Upload a PDF manual. It is rendered, indexed and its fault pages are read into the "
                        "catalog (a few minutes for a long manual). Name the file after the device, e.g. "
                        "`danfoss_fc302_programming.pdf`, so questions naming the device find it.")
            pdf = gr.File(file_types=[".pdf"], label="Manual (PDF)")
            add = gr.Button("Add manual", variant="primary")
            added = gr.Markdown()

        alert.change(from_alert, alert, [text, scope])
        go.click(run, [text, scope, use_catalog], [report, pages, trace])
        text.submit(run, [text, scope, use_catalog], [report, pages, trace])
        find.change(demo.table, find, table)
        again.click(refresh, find, [status, alert, table])
        add.click(upload, pdf, [added, scope, table])
        if hasattr(gr, "Timer"):   # live catalog progress (Gradio 4.40+)
            gr.Timer(10).tick(lambda: demo.status, None, status)
    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public gradio.live link (needed on Kaggle)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-catalog-build", action="store_true", help="don't build missing catalog rows at launch")
    args = ap.parse_args()
    build_ui(Demo(build_catalog=not args.no_catalog_build)).queue().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
