"""OCR pages that have no text layer, so the text-RAG baseline (B1) can search them.

Some PDFs are just page images (scans, slide decks exported as pictures). PyMuPDF finds no
text there, so B1 retrieves nothing and the comparison with the vision systems is unfair.
This fills in the text of those pages with Tesseract OCR and deletes the old text index so it
is rebuilt. The vision systems are unaffected: they read the page images directly.

    apt-get install -y tesseract-ocr && pip install pytesseract
    python scripts/ocr_scanned.py                    # all collections except ViDoRe
    python scripts/ocr_scanned.py --collections mmlongbench manuals
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.config import CFG  # noqa: E402
from docagent.data import PageStore  # noqa: E402


def _ocr(path: str) -> str:
    # One thread per Tesseract: several parallel copies that each spawn threads fight over the CPUs.
    os.environ["OMP_THREAD_LIMIT"] = "1"
    import pytesseract
    from PIL import Image
    img = Image.open(path)
    img.thumbnail((2000, 2000))  # posters and infographics render huge; OCR time grows with pixels
    try:
        return pytesseract.image_to_string(img, timeout=60).strip()
    except RuntimeError:  # pytesseract raises this on timeout; leave the page empty
        return ""


def ocr_collection(name: str, workers: int = 4) -> tuple[int, int]:
    store = PageStore(CFG.pages_dir, name)
    marker = store.root / ".ocr_done"
    if marker.exists():  # already done once; pages where OCR found nothing stay empty
        return -1, 0
    todo = [p for p in store.pages if len(p.text) < CFG.min_text_chars]
    if not todo:
        marker.touch()
        return 0, 0
    print(f"{name}: OCR on {len(todo)} pages without text...", flush=True)
    texts = []
    with ProcessPoolExecutor(workers) as pool:
        for t in pool.map(_ocr, [str(store.image_path(p)) for p in todo], chunksize=4):
            texts.append(t)
            if len(texts) % 10 == 0 or len(texts) == len(todo):
                print(f"  {len(texts)}/{len(todo)} pages", flush=True)
    filled = 0
    for p, t in zip(todo, texts):
        p.scanned = True
        if len(t) > len(p.text):
            p.text = t
            filled += 1
    store.save()
    marker.touch()
    if filled:
        shutil.rmtree(CFG.index_dir / name, ignore_errors=True)  # force the text index to be rebuilt
    return len(todo), filled


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collections", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if shutil.which("tesseract") is None:
        sys.exit("Tesseract is not installed. Run: apt-get install -y tesseract-ocr && pip install pytesseract")

    # ViDoRe ships its own text layer, so OCR there costs hours and adds little.
    names = args.collections or sorted(d.name for d in Path(CFG.pages_dir).iterdir()
                                       if (d / "pages.jsonl").exists() and d.name != "vidore_industrial")
    for name in names:
        n, filled = ocr_collection(name, args.workers)
        if n == -1:
            print(f"{name}: OCR already done")
        elif n:
            print(f"{name}: {n} pages without text, OCR found text on {filled}"
                  + ("; text index will be rebuilt" if filled else ""))
        else:
            print(f"{name}: every page already has text")


if __name__ == "__main__":
    main()
