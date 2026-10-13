"""Restore prepared data from a saved Kaggle version, so a new session doesn't redo downloads.

Kaggle wipes /kaggle/working when a session ends. If you attached the output of a saved version
of this notebook (Add Input > Your Work > this notebook), its files appear somewhere under
/kaggle/input/.../DocAgent/. This copies questions, page stores, text indexes and results back
into the working repo, skipping anything that already exists.

    python scripts/restore_data.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docagent.config import CFG, ROOT  # noqa: E402

PARTS = ["data/questions", "data/pages", "data/indexes", "data/pdfs", "results"]


def find_saved_repo(base: Path = Path("/kaggle/input"), max_depth: int = 5) -> Path | None:
    """Find a DocAgent folder with prepared data, without walking into the thousands of page images."""
    if not base.exists():
        return None
    base_depth = len(base.parts)
    for root, dirs, _ in os.walk(base):
        here = Path(root)
        if here.name == "DocAgent" and (here / "data" / "questions").is_dir():
            return here
        if len(here.parts) - base_depth >= max_depth or here.name in ("pages", "pdfs", "indexes"):
            dirs[:] = []
    return None


def main():
    saved = find_saved_repo()
    if saved is None:
        print("No saved data attached (that's fine on a first run). To reuse data next time: "
              "Save Version, then Add Input > Your Work > this notebook.")
        return
    print(f"Restoring from {saved}")
    for part in PARTS:
        src, dst = saved / part, ROOT / part
        if not src.exists():
            continue
        before = sum(1 for _ in dst.rglob("*")) if dst.exists() else 0
        shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=_copy_if_missing)
        after = sum(1 for _ in dst.rglob("*"))
        print(f"  {part}: {after - before} files restored")
    CFG.ensure_dirs()


def _copy_if_missing(src, dst):
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
    return dst


if __name__ == "__main__":
    main()
