"""Recording the reference voice that XTTS clones.

These clips are the only description of your voice the app ever gets. Room
echo, a fan, or a clipped input end up in every sentence it speaks, so this is
worth doing once in a quiet room rather than redoing later.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from .audio import DeviceInfo
from .config import VOICES_DIR

TARGET_RATE = 24000
RECOMMENDED_SECONDS = 20

# XTTS blends the conditioning across every clip it is given. One long flat
# read teaches it your timbre but only one intonation, so the clone says
# everything the same way. Short clips covering different delivery give it
# much more to work with for the same total recording time.
CLIP_PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "Normal speaking voice",
        "The quick brown fox jumps over the lazy dog. I need bread, milk, "
        "orange juice, and a jar of something sharp.",
    ),
    (
        "Asking a question",
        "How much do you think that actually costs these days? Are you sure "
        "that is the right one?",
    ),
    (
        "Excited",
        "That was incredible, did you see that? I cannot believe it worked on "
        "the first try!",
    ),
    (
        "Calm and slow",
        "She sells sea shells by the sea shore. Take your time, there is no "
        "hurry at all.",
    ),
    (
        "Numbers and dates",
        "Call me at four thirty on Tuesday the twelfth. It was sixty two "
        "pounds, up from forty nine last year.",
    ),
    (
        "Names and places",
        "Sarah and Michael flew from Edinburgh to Vancouver on Thursday. "
        "Ask Priya, Diego or Yusuf when they land.",
    ),
    (
        "A bit faster",
        "Right so the plan is we meet at eight, grab the stuff, drop it off "
        "and head straight back before it gets dark.",
    ),
    (
        "Quieter, relaxed",
        "Yeah, that sounds fine to me. Whenever you are ready, just let me "
        "know and we can sort it out.",
    ),
    (
        "Short replies",
        "Yeah. No, not really. Maybe later. Sure, go ahead. I have no idea. "
        "That is fine.",
    ),
    (
        "Thinking out loud",
        "Hmm, I suppose we could, although I am not completely sure that it "
        "would actually help all that much.",
    ),
    (
        "Reading something out",
        "Order two four seven, dated the third of June, total one hundred and "
        "twelve pounds fifty.",
    ),
    (
        "Tricky sounds",
        "The sixth sick sheikh's sixth sheep is sick. Red lorry, yellow lorry. "
        "Truly rural, thoroughly thorough.",
    ),
)

# Enough clips to be worth saving. All eight is better.
MIN_CLIPS = 3

# Kept for the single-clip path and for tools.
PROMPT = CLIP_PROMPTS[0][1]


class Recorder:
    """Records mono float32 from a device until stopped."""

    def __init__(self, device: DeviceInfo, on_level=None):
        self._device = device
        self._rate = int(device.default_samplerate)
        self._channels = min(device.channels, 2)
        self._on_level = on_level
        self._blocks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        with self._lock:
            self._blocks = []
        self._stream = sd.InputStream(
            device=self._device.index,
            samplerate=self._rate,
            channels=self._channels,
            dtype="float32",
            blocksize=int(self._rate * 0.05),
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, _frames, _time, status):  # noqa: ANN001
        mono = indata[:, 0] if self._channels == 1 else indata.mean(axis=1)
        mono = np.ascontiguousarray(mono, dtype=np.float32)
        if self._on_level:
            self._on_level(float(np.sqrt(np.mean(mono**2))), float(np.abs(mono).max()))
        with self._lock:
            self._blocks.append(mono.copy())

    @property
    def seconds(self) -> float:
        with self._lock:
            return sum(b.size for b in self._blocks) / self._rate

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            if not self._blocks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._blocks)

    @property
    def rate(self) -> int:
        return self._rate


def _trim_silence(audio: np.ndarray, rate: int, thresh_db: float = -45.0) -> np.ndarray:
    if audio.size == 0:
        return audio
    win = max(1, rate // 100)
    trimmed = audio[: (audio.size // win) * win]
    if trimmed.size == 0:
        return audio
    rms = np.sqrt((trimmed.reshape(-1, win) ** 2).mean(axis=1) + 1e-12)
    loud = np.flatnonzero(20 * np.log10(rms) > thresh_db)
    if loud.size == 0:
        return audio
    start = max(0, (loud[0] - 5)) * win
    end = min(audio.size, (loud[-1] + 6) * win)
    return audio[start:end]


def _condition(audio: np.ndarray, rate: int) -> np.ndarray:
    """Trim leading/trailing silence, normalise the peak, resample to 24 kHz."""
    import soxr

    audio = _trim_silence(audio, rate)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 0:
        audio = audio * (0.95 / peak)
    if rate != TARGET_RATE:
        audio = soxr.resample(audio, rate, TARGET_RATE, quality="VHQ")
    return audio.astype(np.float32)


def profile_dir(name: str) -> Path:
    return VOICES_DIR / name


def profile_clips(name: str) -> list[Path]:
    """Every reference recording belonging to a profile.

    Multi-clip profiles live in voices/<name>/NN.wav. The older single-file
    layout, voices/<name>.wav, is still honoured so an existing recording keeps
    working without being re-done.
    """
    directory = profile_dir(name)
    if directory.is_dir():
        clips = sorted(directory.glob("*.wav"))
        if clips:
            return clips
    single = VOICES_DIR / f"{name}.wav"
    return [single] if single.exists() else []


def save_profile(name: str, audio: np.ndarray, rate: int) -> Path:
    """Write a single-clip profile. Kept for the simple path and for tools."""
    return save_profile_clips(name, [(audio, rate)])[0]


def save_profile_clips(name: str, clips: list[tuple[np.ndarray, int]]) -> list[Path]:
    """Write one or more reference clips as voices/<name>/NN.wav.

    XTTS blends the conditioning across every clip it is given, so keeping them
    as separate files rather than concatenating avoids inventing splice points
    that the encoder would read as part of the speech.
    """
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    directory = profile_dir(name)
    if directory.exists():
        for stale in directory.glob("*.wav"):
            stale.unlink()
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for i, (audio, rate) in enumerate(clips, start=1):
        conditioned = _condition(audio, rate)
        if conditioned.size == 0:
            continue
        path = directory / f"{i:02d}.wav"
        sf.write(path, conditioned, TARGET_RATE, subtype="PCM_16")
        written.append(path)

    # A legacy single file of the same name would otherwise shadow nothing but
    # confuse profile listing, and its latents are now stale either way.
    legacy = VOICES_DIR / f"{name}.wav"
    if legacy.exists():
        legacy.unlink()
    cache = VOICES_DIR / f"{name}.latents.pt"
    if cache.exists():
        cache.unlink()
    return written


