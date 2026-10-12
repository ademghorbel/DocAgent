"""Load a Qwen-VL model in 4-bit and ask it questions about text and/or page images.

The same model answers for every system (B1 text RAG, B2 direct VLM, later B3 and DocAgent),
so differences in the results come from retrieval and the agent, not from the model.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

# Reduces fragmentation when prompt sizes vary a lot between questions. Must be set before CUDA starts.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PIL import Image

from .config import CFG


def fit_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale so width*height <= max_pixels. Keeps visual tokens (and GPU memory) per page bounded."""
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    return img.resize((max(28, int(w * scale)), max(28, int(h * scale))), Image.Resampling.LANCZOS)


@dataclass
class GenResult:
    text: str
    latency_s: float
    input_tokens: int
    peak_gpu_gb: float


class VLM:
    def __init__(self, model_id: str = CFG.vlm_model, load_in_4bit: bool = CFG.load_in_4bit,
                 max_pixels: int = CFG.max_pixels_per_page):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.torch = torch
        self.max_pixels = max_pixels
        # T4 (Kaggle/Colab free, compute capability 7.5) has no native bfloat16; Ampere (8.0+) does.
        # is_bf16_supported() can report True on T4 through slow emulation, so check the hardware directly.
        native_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if native_bf16 else torch.float16
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        ) if load_in_4bit else None
        # With two GPUs (Kaggle "T4 x2"), split the layers evenly so each GPU keeps ~10 GB free for
        # attention over many page images. "auto" would pack everything onto GPU 0.
        device_map = "balanced" if torch.cuda.device_count() > 1 else "auto"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, quantization_config=quant, dtype=dtype, device_map=device_map,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit

    def generate(self, prompt: str, images: list[tuple[str, Image.Image]] | None = None,
                 system: str | None = None, max_new_tokens: int = CFG.max_new_tokens) -> GenResult:
        """images: list of (label, PIL image). Each label (e.g. "[manual.pdf p.12]") is placed just before its image."""
        content, pil_images = [], []
        for label, img in images or []:
            content.append({"type": "text", "text": label})
            content.append({"type": "image"})
            pil_images.append(fit_pixels(img.convert("RGB"), self.max_pixels))
        content.append({"type": "text", "text": prompt})
        messages = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []) + \
                   [{"role": "user", "content": content}]

        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat], images=pil_images or None, return_tensors="pt").to(self.model.device)

        cuda = self.torch.cuda.is_available()
        if cuda:
            for i in range(self.torch.cuda.device_count()):
                self.torch.cuda.reset_peak_memory_stats(i)
        start = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        latency = time.perf_counter() - start
        n_in = inputs["input_ids"].shape[1]
        text = self.processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0].strip()
        # Peak summed over GPUs: the real memory cost of this question.
        peak = sum(self.torch.cuda.max_memory_allocated(i) for i in range(self.torch.cuda.device_count())) / 1e9 \
            if cuda else 0.0
        return GenResult(text, latency, int(n_in), round(peak, 2))

    def free_memory(self):
        """Call after an out-of-memory error before retrying with a smaller prompt."""
        import gc
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
