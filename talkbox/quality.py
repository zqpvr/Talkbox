"""Quality presets.

Two dials that cost different things.

Speech quality picks the models. It is paid on every sentence, in both latency
and VRAM, so it is the one worth putting in front of the user.

Cloning quality only changes how much of your recording is turned into speaker
latents. That happens once and is cached to disk, so it costs nothing per
sentence.

VRAM numbers are measured on this machine against a 1.06 GB desktop baseline:
Whisper small 0.64 GB, Whisper medium 1.98 GB, NLLB-600M 0.62 GB via
CTranslate2, NLLB-1.3B about 1.6 GB, XTTS 2.15 GB.

Chunk sizes come from simulating playback rather than from whole-sentence
throughput. What matters is whether the buffer ever empties mid-sentence, not
the average rate. Time to first audio is 19 ms of fixed setup plus 25 ms per
token of lead-in, so the chunk size is close to the whole of it.

On an idle GPU, measured worst-case buffer: chunk 4 runs dry by 0.32 s, chunk 6
keeps 0.23 s, chunk 8 keeps 0.32 s. Six is the fastest safe setting at 200 ms.
Low keeps 8 on purpose: it is the preset for when a game is on the card, and
there a wider margin is worth more than 57 ms.
"""

from __future__ import annotations

from dataclasses import dataclass

CUSTOM = "custom"

NLLB_SMALL = "facebook/nllb-200-distilled-600M"
NLLB_LARGE = "facebook/nllb-200-distilled-1.3B"


@dataclass(frozen=True)
class SpeechPreset:
    key: str
    name: str
    summary: str
    whisper_model: str
    mt_model: str
    tts_chunk_size: int
    tts_repetition_penalty: float
    approx_vram_gb: float
    approx_lag_s: float

    @property
    def label(self) -> str:
        return self.name


SPEECH_PRESETS: tuple[SpeechPreset, ...] = (
    SpeechPreset(
        key="low",
        name="Low",
        summary="Fastest. Misses more words, and leaves room for a game.",
        whisper_model="base",
        mt_model=NLLB_SMALL,
        tts_chunk_size=8,
        tts_repetition_penalty=5.0,
        approx_vram_gb=2.7,
        approx_lag_s=0.6,
    ),
    SpeechPreset(
        key="medium",
        name="Medium",
        summary="Default. Good balance of speed and accuracy for most calls.",
        whisper_model="small",
        mt_model=NLLB_SMALL,
        tts_chunk_size=6,
        tts_repetition_penalty=5.0,
        approx_vram_gb=3.4,
        approx_lag_s=0.6,
    ),
    SpeechPreset(
        key="high",
        name="High",
        summary="Better at hearing you. Helps with accents and background noise.",
        whisper_model="medium",
        mt_model=NLLB_SMALL,
        tts_chunk_size=10,
        tts_repetition_penalty=5.0,
        approx_vram_gb=4.3,
        approx_lag_s=0.9,
    ),
    SpeechPreset(
        key="ultra",
        name="Ultra",
        summary="Better at translating. Best for Russian and Japanese.",
        whisper_model="small",
        mt_model=NLLB_LARGE,
        tts_chunk_size=10,
        tts_repetition_penalty=4.0,
        approx_vram_gb=5.5,
        approx_lag_s=1.0,
    ),
)

BY_KEY = {p.key: p for p in SPEECH_PRESETS}
SPEECH_KEYS = tuple(p.key for p in SPEECH_PRESETS)

# High and Ultra fix different faults, so there is no single "best". Whisper
# medium plus NLLB-1.3B was measured at 7.78 GB of 8.00 GB with 0.21 GB free.
# That leaves nothing for inference and fails with a game running, so the
# ladder never combines them.


@dataclass(frozen=True)
class ClonePreset:
    key: str
    name: str
    summary: str
    gpt_cond_len: int  # -1 means use the entire recording
    max_ref_length: int


# One setting, not a choice. Deriving speaker latents happens once per profile
# and is cached to disk, so there was never a reason to use less of the
# recording than the user actually gave.
FULL_CLONE = ClonePreset(
    key="full",
    name="Full",
    summary="Uses everything you recorded.",
    gpt_cond_len=-1,
    max_ref_length=3600,
)


# Preset keys used by earlier versions. Without this an upgraded config falls
# through to "custom", which quietly keeps whatever model and chunk size the
# old config held and shows no level selected in the dropdown.
LEGACY_KEYS = {
    "fastest": "low",
    "balanced": "medium",
    "clearer": "high",
    "best": "ultra",
}


def speech(cfg) -> SpeechPreset:
    """Effective speech settings for a config."""
    key = getattr(cfg, "speech_quality", "medium")
    key = LEGACY_KEYS.get(key, key)
    preset = BY_KEY.get(key)
    if preset is not None:
        return preset
    return SpeechPreset(
        key=CUSTOM,
        name="Custom",
        summary="Models chosen by hand in Advanced.",
        whisper_model=cfg.whisper_model,
        mt_model=cfg.mt_model,
        tts_chunk_size=cfg.tts_chunk_size,
        tts_repetition_penalty=getattr(cfg, "tts_repetition_penalty", 5.0),
        approx_vram_gb=0.0,
        approx_lag_s=0.0,
    )


def clone(cfg) -> ClonePreset:
    return FULL_CLONE


def voice(cfg) -> tuple[float, int, float]:
    """Map the delivery slider onto XTTS decode parameters.

    Low values keep the decoder close to the most likely continuation: steadier,
    slightly flatter. High values keep more natural variation but occasionally
    read a line oddly. Costs nothing either way.
    """
    e = getattr(cfg, "tts_expressiveness", 0.5)
    e = 0.0 if e < 0.0 else 1.0 if e > 1.0 else float(e)
    return 0.55 + e * 0.35, int(round(25 + e * 45)), 0.80 + e * 0.10
