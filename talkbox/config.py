"""Persisted settings, stored as config.json next to the package."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = APP_DIR / "config.json"
VOICES_DIR = APP_DIR / "voices"
MODELS_DIR = APP_DIR / "models"
TRANSCRIPT_DIR = APP_DIR / "transcripts"


@dataclass
class Config:
    # Audio routing. None means "system default"; these are PortAudio device
    # indices, which are not stable across reboots, so we also store the name
    # and prefer a name match on load.
    input_device: int | None = None
    input_device_name: str = ""
    output_device: int | None = None
    output_device_name: str = ""
    monitor_device: int | None = None
    monitor_device_name: str = ""
    # Off by default. Hearing your own translated voice while you talk is
    # distracting, and it is the setting most likely to cause a feedback loop
    # on speakers. Turn it on from the Audio tab if you want to check output.
    monitor_enabled: bool = False
    # Re-pick microphone, headphones and virtual cable on every launch instead
    # of trusting stored indices, which PortAudio reshuffles between reboots.
    auto_devices: bool = True

    # Direction.
    source_lang: str = "en"
    target_lang: str = "es"

    # One dial. It picks the recogniser and translator, and everything else
    # that trades speed against accuracy follows from it.
    speech_quality: str = "medium"

    # Set from speech_quality at load time; not edited directly any more.
    whisper_model: str = "small"
    compute_type: str = "float16"
    device: str = "cuda"
    mt_model: str = "facebook/nllb-200-distilled-600M"

    # Segmentation. These are the main latency/accuracy knobs.
    vad_threshold: float = 0.5
    # 320 ms splits on comma-length pauses, which produced fragments like a
    # lone "Hello." translating to "- Que pasa?". 400 ms holds clauses together.
    min_silence_ms: int = 400
    preroll_ms: int = 300
    min_speech_ms: int = 250
    max_utterance_s: float = 18.0
    # How long the pause runs when the speculative transcript is still
    # mid-clause. Breaths between clauses measure around 200 ms and thinking
    # pauses around 420 ms, so this has to sit clearly above both to tell
    # either from a real stop. Replaces an earlier rule that shortened the
    # pause to 180 ms past eight seconds and split long sentences in half;
    # see the note at the top of vad.py.
    hold_silence_ms: int = 700
    # When the hard limit is reached with no pause at all, search this far back
    # for the quietest run of frames and cut there instead of mid-word.
    lookback_ms: int = 2500
    pause_window_ms: int = 96

    # Start transcribing a short way into the pause instead of waiting for it
    # to finish. Measured: transcribe + translate is 589 ms, and the 400 ms
    # pause is dead time, so most of that cost can be hidden inside it.
    speculative_recognition: bool = True
    speculate_after_ms: int = 64
    # Do not speculate more often than this within one utterance; a hesitant
    # speaker would otherwise start a transcription at every hesitation.
    speculate_interval_ms: int = 400

    # When a speculative transcript already reads as a finished sentence,
    # ending on trailing punctuation, the pause can stop early. Whisper's
    # punctuation identified every real sentence end in testing and rejected
    # every mid-clause cut but one, where it invented an ending at 0.77
    # confidence, hence the confidence floor.
    early_commit: bool = True
    early_silence_ms: int = 240
    early_commit_confidence: float = 0.85

    # Shown in the transcript but never spoken. A misheard sentence in the
    # user's own voice on a live call is worse than a gap they can repeat.
    min_confidence: float = 0.35

    # "speak" translates and says it out loud in the chosen voice. "dictate"
    # types the sentence into whatever window has focus instead, and never
    # touches the voice model at all.
    mode: str = "speak"
    # Dictation can translate on the way, or write down what was actually
    # said. Off means plain dictation, which is what most people expect the
    # word to mean.
    dictate_translate: bool = False

    # Synthesis.
    # "clone" speaks in the user's recorded voice, "builtin" in one of the
    # speakers shipped inside the XTTS checkpoint. Built-in needs no recording,
    # which makes it the option that works before enrollment and the one to
    # fall back to if a profile goes missing.
    voice_mode: str = "clone"
    builtin_voice: str = "Marcos Rudaski"
    speaker_profile: str = "default"
    # Time to first audio scales with this, and so does the margin against
    # playback underrun. At 8, one test sentence hit a real-time factor of
    # 1.03, meaning generation lost the race with playback. 12 is safe.
    tts_chunk_size: int = 12
    tts_temperature: float = 0.7
    # 0 = steadiest delivery, 1 = closest to your natural variation.
    tts_expressiveness: float = 0.5
    tts_top_k: int = 50
    tts_top_p: float = 0.85
    tts_repetition_penalty: float = 5.0
    tts_speed: float = 1.0
    # XTTS sometimes loops instead of stopping. Backstop only: over twelve
    # runs it never fired (audio 4.2-6.1 s against a 12.8 s cap).
    tts_runaway_factor: float = 2.5
    tts_min_cap_s: float = 5.0
    # On an idle GPU long sentences generate at RTF 0.84-0.94 and the buffer
    # stays positive, but only just: the worst measured margin was 0.11 s. This
    # is jitter insurance for that thin margin, not a fix for a busy GPU.
    # Nothing here helps while a game is using the card; use Low for that.
    prebuffer_below_s: float = 6.0
    prebuffer_fraction: float = 0.10
    prebuffer_max_s: float = 0.6

    # Ignore the microphone while the translated voice is playing, so speakers
    # do not feed the synthesised audio back in and translate it again.
    duplex_guard: bool = True
    # When true the guard is decided from the output device on each Start.
    # Unticking the checkbox in Advanced pins it and clears this.
    duplex_guard_auto: bool = True
    guard_hangover_ms: int = 350
    # Mic level above which "you talked over it" is worth reporting. The gate
    # discards that speech, and doing so without saying anything is worse than
    # the gap itself.
    talkover_rms: float = 0.02

    # Misc.
    coqui_tos_agreed: bool = False
    # First run shows the setup wizard, which downloads the models. Nothing
    # large ships with the source.
    setup_complete: bool = False
    save_transcript: bool = False
    # Off by default. An always-on-top bar over a game is more nuisance than
    # help once the pipeline is working. Turn it on under Advanced.
    show_overlay: bool = False



    @classmethod
    def load(cls) -> "Config":
        cfg = cls()
        if CONFIG_PATH.exists():
            try:
                raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return cfg
            known = {f.name for f in fields(cls)}
            for key, value in raw.items():
                if key in known:
                    setattr(cfg, key, value)

            # Rename presets from earlier versions, then let the current
            # defaults win for anything the old file pinned. Keeping the old
            # chunk size, for instance, would silently hold latency back.
            from .quality import LEGACY_KEYS

            if cfg.speech_quality in LEGACY_KEYS:
                cfg.speech_quality = LEGACY_KEYS[cfg.speech_quality]
                fresh = cls()
                for tuned in ("tts_chunk_size", "min_silence_ms", "max_utterance_s"):
                    setattr(cfg, tuned, getattr(fresh, tuned))
        return cfg

    def save(self) -> None:
        CONFIG_PATH.write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )
