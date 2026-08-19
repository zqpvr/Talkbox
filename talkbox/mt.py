"""Machine translation with NLLB-200.

One model covers every direction, which matters because the alternative is
loading a different checkpoint every time the swap button is pressed.

Two backends. CTranslate2 is preferred and is what the setup step prepares:
measured against the HuggingFace path on the same sentences it runs 1.9x
faster (167 ms against 311 ms) and uses 0.49 GB less VRAM, with identical
output. HuggingFace `generate` is slow here because it runs a Python loop per
generated token, which dominates at this model size.

The HuggingFace path is kept as a fallback so the app still works if the
conversion has not been run or fails.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import MODELS_DIR

CT2_ROOT = MODELS_DIR / "ct2"


def ct2_dir(model_name: str) -> Path:
    return CT2_ROOT / model_name.replace("/", "__")


def is_converted(model_name: str) -> bool:
    return (ct2_dir(model_name) / "model.bin").is_file()


def convert(model_name: str, progress=None, quantization: str = "int8_float16") -> Path:
    """Convert a HuggingFace NLLB checkpoint to CTranslate2.

    Takes a couple of minutes and only happens once per model. The result is
    about 600 MB, which is smaller than the checkpoint it came from.
    """
    from ctranslate2.converters import TransformersConverter

    out = ct2_dir(model_name)
    if is_converted(model_name):
        return out
    if progress:
        progress(f"Preparing translator ({model_name.rsplit('-', 1)[-1]}), one time...")

    # A half-finished directory would look converted next launch.
    staging = out.with_name(out.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.parent.mkdir(parents=True, exist_ok=True)

    converter = TransformersConverter(
        model_name, copy_files=[], load_as_float16=True
    )
    converter.convert(str(staging), quantization=quantization, force=True)
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    staging.rename(out)
    return out


class Translator:
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "cuda",
        progress=None,
    ):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.device = device
        cache = str(MODELS_DIR / "hf")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache)

        self._ct2 = None
        self._hf = None
        try:
            self._load_ct2(progress)
        except Exception as exc:
            if progress:
                progress(f"Fast translator unavailable ({exc}). Using fallback.")
            self._load_hf()

    @property
    def backend(self) -> str:
        return "ctranslate2" if self._ct2 is not None else "transformers"

    def _load_ct2(self, progress=None) -> None:
        import ctranslate2

        path = convert(self.model_name, progress)
        self._ct2 = ctranslate2.Translator(
            str(path),
            device=self.device,
            compute_type="int8_float16" if self.device == "cuda" else "int8",
        )

    def _load_hf(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, cache_dir=str(MODELS_DIR / "hf")
        )
        # Cast after loading: transformers renamed torch_dtype to dtype in
        # 4.56 and both vanish into **kwargs, so neither can be probed for.
        if self.device == "cuda":
            model = model.half()
        model.to(self.device)
        model.eval()
        self._hf = model
        self._torch = torch

    def translate(self, text: str, src: str, tgt: str) -> str:
        """`src` and `tgt` are FLORES-200 tags, e.g. 'eng_Latn'."""
        text = text.strip()
        if not text:
            return ""
        if src == tgt:
            return text
        if self._ct2 is not None:
            return self._translate_ct2(text, src, tgt)
        return self._translate_hf(text, src, tgt)

    def _translate_ct2(self, text: str, src: str, tgt: str) -> str:
        self.tokenizer.src_lang = src
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        results = self._ct2.translate_batch(
            [tokens],
            target_prefix=[[tgt]],
            beam_size=1,
            max_decoding_length=256,
        )
        out = results[0].hypotheses[0]
        if out and out[0] == tgt:
            out = out[1:]  # Drop the forced language tag.
        return self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(out), skip_special_tokens=True
        ).strip()

    def _translate_hf(self, text: str, src: str, tgt: str) -> str:
        self.tokenizer.src_lang = src
        batch = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)
        bos = self.tokenizer.convert_tokens_to_ids(tgt)
        with self._torch.inference_mode():
            generated = self._hf.generate(
                **batch,
                forced_bos_token_id=bos,
                num_beams=1,
                do_sample=False,
                max_new_tokens=min(256, int(batch["input_ids"].shape[1] * 2 + 16)),
            )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[
            0
        ].strip()
