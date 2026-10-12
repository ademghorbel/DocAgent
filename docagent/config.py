"""Central settings. Override any value with an environment variable of the same name."""
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name, default, cast=str):
    value = os.environ.get(name)
    return cast(value) if value is not None else default


ROOT = Path(_env("DOCAGENT_ROOT", Path(__file__).resolve().parents[1]))


@dataclass
class Config:
    # Folders
    data_dir: Path = ROOT / "data"
    pages_dir: Path = ROOT / "data" / "pages"          # rendered page images + text, one folder per collection
    pdf_dir: Path = ROOT / "data" / "pdfs"              # downloaded PDFs (git-ignored)
    index_dir: Path = ROOT / "data" / "indexes"         # FAISS indexes (git-ignored)
    results_dir: Path = ROOT / "results"

    # Rendering
    dpi: int = _env("DOCAGENT_DPI", 144, int)           # 144 dpi keeps an A4 page around 1190x1684 px
    min_text_chars: int = 50                            # below this, a page is treated as scanned

    # Text RAG (B1)
    embed_model: str = _env("DOCAGENT_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    chunk_chars: int = 1200
    chunk_overlap: int = 200
    top_k: int = _env("DOCAGENT_TOP_K", 5, int)

    # VLM, used by every system so the comparison is fair
    vlm_model: str = _env("DOCAGENT_VLM", "Qwen/Qwen2.5-VL-7B-Instruct")
    load_in_4bit: bool = _env("DOCAGENT_4BIT", "1") == "1"
    max_new_tokens: int = _env("DOCAGENT_MAX_NEW_TOKENS", 256, int)
    # Qwen-VL turns each 28x28 pixel block into one visual token; this caps tokens per page image.
    max_pixels_per_page: int = _env("DOCAGENT_MAX_PIXELS", 1024 * 28 * 28, int)

    # Direct VLM (B2)
    direct_vlm_pages: int = _env("DOCAGENT_DIRECT_PAGES", 8, int)

    seed: int = 42
    extra: dict = field(default_factory=dict)

    def ensure_dirs(self):
        for d in (self.data_dir, self.pages_dir, self.pdf_dir, self.index_dir, self.results_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


CFG = Config()
