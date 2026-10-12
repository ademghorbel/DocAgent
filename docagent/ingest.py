"""PDF -> page images + page text, written into a PageStore.

DocAgent itself only needs the images. The text layer is kept for the text-RAG baseline (B1)
and to flag scanned pages, which have no text layer and need OCR for B1 only.

    python -m docagent.ingest --collection manuals --pdfs data/pdfs/manuals
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf

from .config import CFG
from .data import Page, PageStore, safe_name

# Many real PDFs have small defects (e.g. "could not parse color space"). MuPDF still renders the
# page fine but prints a line per defect; hide that noise so real errors stay visible.
pymupdf.TOOLS.mupdf_display_errors(False)


def ocr_page(png_path: Path) -> str:
    """Optional OCR for scanned pages (B1 only). Needs `pip install pytesseract` + the tesseract binary."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:  # keep ingestion working without OCR installed
        raise RuntimeError("OCR requested but pytesseract is not installed") from e
    return pytesseract.image_to_string(Image.open(png_path))


def ingest_pdf(pdf_path: Path, store: PageStore, dpi: int = CFG.dpi, doc_id: str | None = None,
               use_ocr: bool = False, min_text_chars: int = CFG.min_text_chars) -> dict:
    pdf_path = Path(pdf_path)
    doc_id = doc_id or pdf_path.name
    folder = store.root / safe_name(doc_id)
    folder.mkdir(parents=True, exist_ok=True)
    stats = {"doc_id": doc_id, "pages": 0, "scanned": 0, "ocr": 0}

    with pymupdf.open(pdf_path) as doc:
        for i, pg in enumerate(doc, start=1):
            if store.get(doc_id, i):          # already ingested, skip (makes reruns cheap)
                stats["pages"] += 1
                continue
            img_rel = f"{safe_name(doc_id)}/p{i:04d}.png"
            pg.get_pixmap(dpi=dpi).save(store.root / img_rel)
            text = pg.get_text("text").strip()
            scanned = len(text) < min_text_chars
            if scanned:
                stats["scanned"] += 1
                if use_ocr:
                    text = ocr_page(store.root / img_rel).strip()
                    stats["ocr"] += 1
            store.add(Page(store.collection, doc_id, i, img_rel, text, scanned))
            stats["pages"] += 1
    return stats


def ingest_folder(pdf_dir: Path, collection: str, use_ocr: bool = False) -> list[dict]:
    store = PageStore(CFG.pages_dir, collection)
    store.root.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        s = ingest_pdf(pdf, store, use_ocr=use_ocr)
        all_stats.append(s)
        flag = f"  ({s['scanned']} scanned{', OCR applied' if s['ocr'] else ''})" if s["scanned"] else ""
        print(f"{s['doc_id']}: {s['pages']} pages{flag}")
    store.save()
    print(f"Collection '{collection}': {len(store)} pages in {store.root}")
    return all_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--pdfs", required=True, type=Path)
    ap.add_argument("--ocr", action="store_true", help="OCR scanned pages so text RAG can see them")
    args = ap.parse_args()
    ingest_folder(args.pdfs, args.collection, use_ocr=args.ocr)


if __name__ == "__main__":
    main()
