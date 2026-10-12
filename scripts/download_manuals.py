"""Download the manuals listed in data/manuals.csv, then ingest them.

    python scripts/download_manuals.py
    python scripts/download_manuals.py --ocr                       # also OCR scanned pages for the text baseline
    python scripts/download_manuals.py --from /kaggle/input/manuals  # add PDFs you uploaded by hand

Some manufacturer sites block scripts (HTTP 403). Those manuals are skipped with a message;
download them in your browser, upload them to Kaggle, and rerun with --from.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.config import CFG  # noqa: E402
from docagent.ingest import ingest_folder  # noqa: E402

COLLECTION = "manuals"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/128.0.0.0 Safari/537.36",
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def rows(csv_path: Path):
    with csv_path.open(encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    yield from csv.DictReader(lines)


def fetch_urllib(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=90) as resp:
        return resp.read()


def fetch_curl(url: str) -> bytes:
    cmd = ["curl", "-sSL", "--fail", "--max-time", "120"]
    for k, v in HEADERS.items():
        cmd += ["-H", f"{k}: {v}"]
    return subprocess.run(cmd + [url], capture_output=True, check=True).stdout


def download(url: str) -> tuple[bytes | None, str]:
    errors = []
    for fetch in (fetch_urllib, fetch_curl):
        try:
            data = fetch(url)
            if data.startswith(b"%PDF"):
                return data, ""
            errors.append(f"{fetch.__name__}: got a web page, not a PDF")
        except Exception as e:  # noqa: BLE001  (any network error: try the next method, then skip)
            msg = e.stderr.decode(errors="ignore").strip() if isinstance(e, subprocess.CalledProcessError) else str(e)
            errors.append(f"{fetch.__name__}: {msg[:120]}")
    return None, " | ".join(errors)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=CFG.data_dir / "manuals.csv")
    ap.add_argument("--from", dest="from_dir", type=Path, help="folder of PDFs downloaded by hand")
    ap.add_argument("--ocr", action="store_true")
    args = ap.parse_args()

    out = CFG.pdf_dir / COLLECTION
    out.mkdir(parents=True, exist_ok=True)

    if args.from_dir:
        found = sorted(args.from_dir.rglob("*.pdf"))
        if not found:
            sys.exit(f"No PDFs found under {args.from_dir}. Run `!find /kaggle/input -name '*.pdf'` to see where they are.")
        for pdf in found:
            shutil.copy(pdf, out / pdf.name)
            print(f"added {pdf.name} (uploaded by hand)")

    skipped = []
    for r in rows(args.csv):
        dest = out / r["doc_id"]
        if dest.exists():
            print(f"have  {dest.name}")
            continue
        data, err = download(r["url"])
        if data is None:
            skipped.append(r)
            print(f"SKIP  {r['doc_id']}: {err}")
            continue
        dest.write_bytes(data)
        print(f"got   {dest.name} ({len(data) / 1e6:.1f} MB)")

    if skipped:
        print(f"\n{len(skipped)} manual(s) blocked the download. Get them by hand:")
        for r in skipped:
            print(f"  {r['doc_id']}  <-  {r['url']}")
        print("Open each link in your browser, save the PDF under the name on the left,\n"
              "upload them to Kaggle (Add Input > Upload > New dataset), then rerun with\n"
              "  --from /kaggle/input/<your-dataset-name>\n")

    if not any(out.glob("*.pdf")):
        sys.exit("No manuals downloaded yet.")
    ingest_folder(out, COLLECTION, use_ocr=args.ocr)


if __name__ == "__main__":
    main()
