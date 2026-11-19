"""Build the fault catalog: the VLM reads each manual's fault pages and writes one row per code.

    python scripts/build_catalog.py                    # every manual not catalogued yet
    python scripts/build_catalog.py --docs abb_acs580_firmware.pdf --redo
    python scripts/build_catalog.py --pages            # only list the fault pages found (no GPU)

Rows go to data/catalog/faults.jsonl (DOCAGENT_CATALOG to change). Saved after each manual, so a
stopped run resumes. The demo builds missing rows by itself too; this script is for doing it ahead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.catalog import Catalog, build, device_name, fault_pages  # noqa: E402
from docagent.config import CFG  # noqa: E402
from docagent.data import PageStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="manuals")
    ap.add_argument("--docs", nargs="*", help="only these manuals (default: all)")
    ap.add_argument("--redo", action="store_true", help="read these manuals again even if catalogued")
    ap.add_argument("--pages", action="store_true", help="only print the fault pages each manual would use")
    args = ap.parse_args()

    store = PageStore(CFG.pages_dir, args.collection)
    docs = args.docs or store.doc_ids()
    if args.pages:
        for d in docs:
            print(f"{device_name(d)}: {fault_pages(store, d)}")
        return
    catalog = Catalog()
    if args.redo:
        catalog.done -= set(docs)
    from docagent.vlm import VLM
    build(store, VLM(), catalog, docs)
    print(f"Catalog: {len(catalog)} codes from {len(catalog.done)} manuals in {catalog.path}")


if __name__ == "__main__":
    main()
