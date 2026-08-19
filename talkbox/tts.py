"""Voice-cloning text to speech with XTTS-v2.

XTTS is the piece that makes the whole thing work as asked: it is zero-shot, so
a single ~20 second recording of your voice is enough, and its speaker
conditioning is language-independent, so the same timbre carries into all
seventeen output languages. It also exposes a true streaming interface, which
means the first audio leaves the GPU long before the sentence has finished
generating.

The model is pulled straight from the Hugging Face repo rather than through
Coqui's ModelManager. ModelManager wants to prompt on stdin for license
agreement, which does not work from a GUI process; the agreement is instead
collected once during setup and recorded in config.json.

Licence note: XTTS-v2 is released under the Coqui Public Model License 1.0,
which permits non-commercial use only.
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import MODELS_DIR, VOICES_DIR

HF_REPO = "coqui/XTTS-v2"
SAMPLE_RATE = 24000

# XTTS ships a bank of 58 studio speakers in speakers_xtts.pth, each stored as
# exactly the pair of tensors get_conditioning_latents produces from a
# recording. A built-in voice is therefore a drop-in for a cloned one: same
# model, same synthesis path, same latency, and no enrollment step at all.
# Their conditioning is language-independent in the same way a clone's is, so
# one speaker covers all seventeen output languages.
BUILTIN_DEFAULT = "Marcos Rudaski"

# The speaker bank is fixed for XTTS-v2, so the names are listed here as well
# as read from the checkpoint. That lets the voice list populate before the
# model has been downloaded or loaded; list_builtin() below reports what the
# checkpoint on this machine actually contains.
BUILTIN_VOICES: tuple[str, ...] = (
    "Aaron Dreschner", "Abrahan Mack", "Adde Michal", "Alexandra Hisakawa",
    "Alison Dietlinde", "Alma María", "Ana Florence", "Andrew Chipper",
    "Annmarie Nele", "Asya Anara", "Badr Odhiambo", "Baldur Sanjin",
    "Barbora MacLean", "Brenda Stern", "Camilla Holmström",
    "Chandra MacFarland", "Claribel Dervla", "Craig Gutsy", "Daisy Studious",
    "Damien Black", "Damjan Chapman", "Dionisio Schuyler", "Eugenio Mataracı",
    "Ferran Simen", "Filip Traverse", "Gilberto Mathias", "Gitta Nikolina",
    "Gracie Wise", "Henriette Usha", "Ige Behringer", "Ilkin Urbano",
    "Kazuhiko Atallah", "Kumar Dahl", "Lidiya Szekeres", "Lilya Stainthorpe",
    "Ludvig Milivoj", "Luis Moray", "Maja Ruoho", "Marcos Rudaski",
    "Narelle Moon", "Nova Hogarth", "Rosemary Okafor", "Royston Min",
    "Sofia Hellen", "Suad Qasim", "Szofi Granger", "Tammie Ema", "Tammy Grit",
    "Tanja Adelina", "Torcull Diarmuid", "Uta Obando", "Viktor Eka",
    "Viktor Menelaos", "Vjollca Johnnie", "Wulf Carlevaro", "Xavier Hayasaka",
    "Zacharie Aimilios", "Zofija Kendrick",
)



def _allow_xtts_unpickling() -> None:
    """torch>=2.6 defaults torch.load to weights_only, which rejects the
    dataclasses inside the XTTS checkpoint. Allow-list just those."""
    import torch

    if not hasattr(torch.serialization, "add_safe_globals"):
        return
    safe = []
    try:
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig

        safe = [XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig, Xtts]
    except ImportError:
        return
    try:
        torch.serialization.add_safe_globals(safe)
    except Exception:
        pass


def download_model(progress=None) -> Path:
    from huggingface_hub import snapshot_download

    if progress:
        progress("Downloading XTTS-v2 (~1.9 GB, first run only)...")
    path = snapshot_download(
        repo_id=HF_REPO,
        cache_dir=str(MODELS_DIR / "hf"),
        allow_patterns=[
            "config.json",
            "model.pth",
            "vocab.json",
            "speakers_xtts.pth",
            "dvae.pth",
            "mel_stats.pth",
        ],
    )
    return Path(path)


# XTTS splits text over 250 characters itself, but that path imports spaCy and
# raises ImportError without it, for every language and not only the one named
# in the message. Installing spaCy plus the Japanese tokenizer and its
# dictionaries is a few hundred megabytes to do a job we can do here, so we
# split first and hand XTTS pieces it will never try to split again.
#
# The limits are XTTS's own, from XttsTokenizer.char_limits. They matter: past
# them the model truncates the audio, and its 250-character splitter is far too
# generous for Japanese (71), Korean (95) and Chinese (82). Splitting on a
# single global size would silently cut those languages off mid-sentence.
CHAR_LIMITS = {
    "en": 250,
    "de": 253,
    "fr": 273,
    "es": 239,
    "it": 213,
    "pt": 203,
    "pl": 224,
    "zh": 82,
    "zh-cn": 82,
    "ar": 166,
    "cs": 186,
    "ru": 182,
    "nl": 251,
    "tr": 226,
    "ja": 71,
    "hu": 224,
    "ko": 95,
    "hi": 150,
}
# A little under the limit, so punctuation and spacing added by the tokenizer
# cannot push a piece over.
SAFETY = 0.9


def chunk_limit(language: str) -> int:
    limit = CHAR_LIMITS.get(language, CHAR_LIMITS.get(language.split("-")[0], 250))
    return max(40, int(limit * SAFETY))

# Sentence enders across the seventeen supported languages, including the CJK
# full-width forms, the Arabic question mark, and the Devanagari danda.
_SENTENCE_END = re.compile(r"(?<=[.!?;:\u3002\uff01\uff1f\uff1b\u061f\u06d4\u0964])\s*")
# Languages written without spaces, where wrapping has to fall back to counting
# characters rather than looking for a word boundary.
_NO_SPACES = ("zh", "ja")


def split_for_synthesis(text: str, language: str, limit: int | None = None):
    """Break text into pieces XTTS can speak in one go.

    Splits on sentence punctuation, then packs consecutive sentences back
    together up to the language's limit so short sentences are not spoken with
    a gap between every one of them.
    """
    if limit is None:
        limit = chunk_limit(language)
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= limit:
            pieces.append(sentence)
            continue
        # One sentence longer than the limit. Wrap on spaces where the language
        # has them, otherwise cut on character count.
        if language.split("-")[0] in _NO_SPACES:
            pieces.extend(
                sentence[i : i + limit] for i in range(0, len(sentence), limit)
            )
        else:
            pieces.extend(textwrap.wrap(sentence, width=limit, break_long_words=False))

    packed: list[str] = []
    for piece in pieces:
        if packed and len(packed[-1]) + 1 + len(piece) <= limit:
            packed[-1] = f"{packed[-1]} {piece}"
        else:
            packed.append(piece)
    return packed


class LicenceNotAccepted(RuntimeError):
    """Raised when XTTS is requested before its licence has been accepted."""


class VoiceCloner:
    def __init__(self, device: str = "cuda", progress=None, tos_agreed: bool = False):
        if not tos_agreed:
            raise LicenceNotAccepted(
                "The XTTS-v2 model licence (Coqui Public Model License, "
                "non-commercial) has not been accepted. Run setup.ps1, or "
                "accept the prompt shown at startup."
            )
        os.environ["COQUI_TOS_AGREED"] = "1"
        _allow_xtts_unpickling()

        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        self._torch = torch
        self.device = device

        model_dir = download_model(progress)
        if progress:
            progress("Loading XTTS-v2 onto the GPU...")

        config = XttsConfig()
        config.load_json(str(model_dir / "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config, checkpoint_dir=str(model_dir), use_deepspeed=False
        )
        if device == "cuda":
            self.model.cuda()
        self.model.eval()

        self._latents: tuple | None = None
        self._profile: str | None = None

    # -- speaker conditioning -------------------------------------------------

    @staticmethod
    def profile_wav(name: str) -> Path:
        return VOICES_DIR / f"{name}.wav"

    @staticmethod
    def profile_cache(name: str) -> Path:
        return VOICES_DIR / f"{name}.latents.pt"

    @staticmethod
    def profile_clips(name: str) -> list[Path]:
        from .enroll import profile_clips

        return profile_clips(name)

    # -- built-in speakers ----------------------------------------------------

    def list_builtin(self) -> list[str]:
        manager = getattr(self.model, "speaker_manager", None)
        speakers = getattr(manager, "speakers", None) or {}
        return sorted(speakers.keys())

    def load_builtin(self, name: str = BUILTIN_DEFAULT) -> None:
        """Use one of XTTS's own speakers instead of a recorded voice.

        No latents are computed and nothing is cached, because the checkpoint
        already stores them. Loading is a tensor move, so switching voices is
        effectively instant.
        """
        manager = getattr(self.model, "speaker_manager", None)
        speakers = getattr(manager, "speakers", None) or {}
        if not speakers:
            raise RuntimeError(
                "This XTTS checkpoint has no built-in speakers. Delete the "
                "models/hf folder and let setup download it again."
            )
        entry = speakers.get(name)
        if entry is None:
            # A config naming a speaker this checkpoint does not have should
            # fall back to speaking rather than fail to start.
            name = BUILTIN_DEFAULT if BUILTIN_DEFAULT in speakers else sorted(speakers)[0]
            entry = speakers[name]
        gpt = entry["gpt_cond_latent"]
        spk = entry["speaker_embedding"]
        self._latents = (gpt.to(self.device), spk.to(self.device))
        self._profile = f"builtin:{name}"

    @staticmethod
    def list_profiles() -> list[str]:
        """Profiles are either voices/<name>.wav or voices/<name>/*.wav."""
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        names = {p.stem for p in VOICES_DIR.glob("*.wav")}
        names |= {
            d.name for d in VOICES_DIR.iterdir() if d.is_dir() and any(d.glob("*.wav"))
        }
        return sorted(names)

    def load_profile(
        self, name: str, gpt_cond_len: int = -1, max_ref_length: int = 3600
    ) -> None:
        """Compute (or reuse) the speaker latents for a recorded voice.

        `gpt_cond_len=-1` tells XTTS to use the whole recording rather than
        the first 30 seconds, and `max_ref_length` has to be larger than the
        longest clip or each one is truncated before it is read.

        The cache is keyed on these parameters as well as the recording's
        timestamp, so changing them recomputes instead of silently reusing
        latents built at the old setting.
        """
        clips = self.profile_clips(name)
        if not clips:
            raise FileNotFoundError(
                f"No voice profile named '{name}'. Record one first."
            )

        stamp = (int(gpt_cond_len), int(max_ref_length), len(clips))
        newest = max(c.stat().st_mtime for c in clips)
        cache = self.profile_cache(name)
        blob = None
        if cache.exists() and cache.stat().st_mtime >= newest:
            try:
                blob = self._torch.load(
                    cache, map_location=self.device, weights_only=True
                )
            except Exception:
                blob = None
            if blob is not None and tuple(blob.get("params", ())) != stamp:
                blob = None  # Cloned at different settings; recompute.

        if blob is not None:
            self._latents = (blob["gpt_cond_latent"], blob["speaker_embedding"])
        else:
            # XTTS averages the conditioning across every clip supplied, which
            # is what gives a multi-clip profile its wider prosodic range.
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=[str(c) for c in clips],
                gpt_cond_len=gpt_cond_len,
                max_ref_length=max_ref_length,
            )
            self._torch.save(
                {
                    "gpt_cond_latent": gpt_cond_latent.cpu(),
                    "speaker_embedding": speaker_embedding.cpu(),
                    "params": stamp,
                },
                cache,
            )
            self._latents = (gpt_cond_latent, speaker_embedding)

        gpt, spk = self._latents
        self._latents = (gpt.to(self.device), spk.to(self.device))
        self._profile = name

    def invalidate(self, name: str) -> None:
        cache = self.profile_cache(name)
        if cache.exists():
            cache.unlink()
        if self._profile == name:
            self._profile = None
            self._latents = None

    # -- synthesis ------------------------------------------------------------

    def stream(
        self,
        text: str,
        language: str,
        chunk_size: int = 12,
        temperature: float = 0.7,
        speed: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.85,
        repetition_penalty: float = 5.0,
    ) -> Iterator[np.ndarray]:
        """Yield float32 mono chunks at 24 kHz as they are generated."""
        if self._latents is None:
            raise RuntimeError("No voice profile loaded.")
        text = text.strip()
        if not text:
            return

        gpt_cond_latent, speaker_embedding = self._latents
        kwargs = dict(
            temperature=temperature,
            stream_chunk_size=chunk_size,
            # Off on purpose: we already split the text above. Leaving it on
            # makes XTTS import spaCy for anything past 250 characters.
            enable_text_splitting=False,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
        )
        if abs(speed - 1.0) > 1e-3:
            kwargs["speed"] = speed

        with self._torch.inference_mode():
            for part in split_for_synthesis(text, language):
                try:
                    chunks = self.model.inference_stream(
                        part, language, gpt_cond_latent, speaker_embedding, **kwargs
                    )
                except TypeError:
                    kwargs.pop("speed", None)
                    chunks = self.model.inference_stream(
                        part, language, gpt_cond_latent, speaker_embedding, **kwargs
                    )
                for chunk in chunks:
                    yield chunk.squeeze().detach().cpu().numpy().astype(np.float32)


def _builtin_labels(names):
    """First names for the voice list, since the surnames carry no meaning.

    Two of the fifty-eight share a first name, so those keep an initial to
    tell them apart. Worked out from the list rather than hardcoded, so it
    stays right if the checkpoint's speaker set ever changes.
    """
    by_first: dict[str, list[str]] = {}
    for full in names:
        by_first.setdefault(full.split()[0], []).append(full)
    labels: dict[str, str] = {}
    for first, group in by_first.items():
        for full in group:
            if len(group) == 1:
                labels[full] = first
            else:
                labels[full] = f"{first} {full.split()[-1][0]}."
    return labels


BUILTIN_LABELS = _builtin_labels(BUILTIN_VOICES)

# Ordered by what the user actually sees, not by surname.
BUILTIN_ORDER: tuple[str, ...] = tuple(
    sorted(BUILTIN_VOICES, key=lambda full: BUILTIN_LABELS[full])
)

