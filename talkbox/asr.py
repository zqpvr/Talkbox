"""Speech recognition via faster-whisper (CTranslate2).

Decoding is deliberately greedy. Beam search buys a small accuracy gain for a
large latency cost, and in a live call the extra 200-400 ms is far more
noticeable than the occasional wrong word, which the translation model tends to
absorb anyway.
"""

from __future__ import annotations

import re

import numpy as np

from .config import MODELS_DIR

# Whisper reliably invents these when handed near-silence or a stray click.
# They are safe to drop wholesale because nobody says them mid-call either.
_HALLUCINATIONS = {
    "thank you.",
    "thanks for watching!",
    "thank you for watching.",
    "you",
    "bye.",
    "subtitles by the amara.org community",
    "please subscribe.",
    "amara.org",
    "ご視聴ありがとうございました",
}

_PUNCT_ONLY = re.compile(r"^[\s\.,!?\-–—…\"'`~]*$")


class Transcriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        from faster_whisper import WhisperModel

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(MODELS_DIR / "whisper"),
        )

    def transcribe(
        self, audio: np.ndarray, language: str | None
    ) -> tuple[str, str, float]:
        """Return (text, detected_language, confidence).

        Detection comes free with the decode pass, so auto mode costs nothing
        extra compared to running detect_language() separately first.

        Confidence is the duration-weighted mean of Whisper's per-segment
        `avg_logprob`, mapped onto 0..1. It is not a probability, but it
        separates "heard that clearly" from "guessed" well enough to decide
        whether a sentence is safe to speak talkbox into someone's call.
        """
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            no_speech_threshold=0.6,
        )

        parts: list[str] = []
        weighted, total = 0.0, 0.0
        for seg in segments:
            if seg.no_speech_prob > 0.7 and seg.avg_logprob < -0.7:
                continue
            parts.append(seg.text)
            span = max(0.01, float(seg.end) - float(seg.start))
            weighted += float(seg.avg_logprob) * span
            total += span

        text = " ".join(p.strip() for p in parts).strip()
        text = re.sub(r"\s+", " ", text)
        detected = getattr(info, "language", language) or (language or "en")

        # avg_logprob runs roughly -1.2 (garbage) to -0.1 (clean). Map that
        # span onto 0..1 rather than inventing a calibrated probability.
        mean_logprob = (weighted / total) if total else -1.2
        confidence = (mean_logprob + 1.2) / 1.1
        confidence = 0.0 if confidence < 0 else 1.0 if confidence > 1 else confidence

        if not text or _PUNCT_ONLY.match(text):
            return "", detected, 0.0
        if text.strip().lower() in _HALLUCINATIONS:
            return "", detected, 0.0
        if getattr(info, "language_probability", 1.0) < 0.35:
            return "", detected, 0.0
        return text, detected, confidence
