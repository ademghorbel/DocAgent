"""Download the manuals listed in data/manuals.csv, then ingest them.

    python scripts/download_manuals.py
    python scripts/download_manuals.py --ocr     # also OCR scanned pages for the text baseline
"""
from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.config import CFG  # noqa: E402
from docagent.ingest import ingest_folder  # noqa: E402

COLLECTION = "manuals"
UA = "Mozilla/5.0 (DocAgent research download)"


def rows(csv_path: Path):
    with csv_path.open(encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    yield from csv.DictReader(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=CFG.data_dir / "manuals.csv")
    ap.add_argument("--ocr", action="store_true")
    args = ap.parse_args()

    out = CFG.pdf_dir / COLLECTION
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in rows(args.csv):
        dest = out / r["doc_id"]
        n += 1
        if dest.exists():
            print(f"have  {dest.name}")
            continue
        req = urllib.request.Request(r["url"], headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if not data.startswith(b"%PDF"):
            print(f"SKIP  {r['doc_id']}: the link did not return a PDF (maybe a login or landing page)")
            continue
        dest.write_bytes(data)
        print(f"got   {dest.name} ({len(data) / 1e6:.1f} MB)")
    if n == 0:
        sys.exit("No rows in manuals.csv yet: add your manual links first.")
    ingest_folder(out, COLLECTION, use_ocr=args.ocr)


if __name__ == "__main__":
    main()
