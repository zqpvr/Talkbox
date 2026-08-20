"""End-to-end pipeline check that needs no microphone.

Windows SAPI is used to fabricate two WAV files: a long one that stands in for
a recorded reference voice, and a short one that stands in for something you
just said. Those are pushed through the real Transcriber, Translator and
VoiceCloner, and the result is written out as a WAV you can listen to.

This exercises every model and every code path except audio device I/O and the
GUI, so it is the fastest way to tell whether an install actually works and
what the per-stage latency looks like on this machine.

    .venv\\Scripts\\python.exe tools\\smoke_test.py
    .venv\\Scripts\\python.exe tools\\smoke_test.py --target Japanese
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from talkbox.runtime import ensure_cuda_dlls  # noqa: E402

ensure_cuda_dlls()

# Windows consoles default to cp1252, which cannot print Japanese, Chinese,
# Korean, Arabic or Hindi output. Without this the tool crashes on its own
# result line for exactly the languages worth checking.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from talkbox import languages  # noqa: E402

OUT_DIR = ROOT / "tools" / "_smoke"

REFERENCE_TEXT = (
    "The quick brown fox jumps over the lazy dog while five wizards vex the "
    "gnome by the quay. How much would you say a rough, thoughtful measurement "
    "really costs these days? I need bread, milk, orange juice, eleven yellow "
    "pencils, and a jar of something sharp. She sells sea shells by the sea "
    "shore, and the shells she sells are surely seashells. This recording "
    "exists only so the system has a voice to imitate."
)

UTTERANCE_TEXT = (
    "Hello, can you hear me clearly? I would like to talk about the project "
    "schedule for next week."
)

# A long passage as well. XTTS only reaches its own text splitter past 250
# characters, and that path raises ImportError without spaCy, so a short test
# sentence never exercised it. Per-language character limits are far lower
# still (Japanese 71), so long text is where truncation shows up too.
LONG_TEXT = (
    "So what I wanted to talk about today is the schedule for next week, "
    "because I think we need to move the meeting to Wednesday morning instead "
    "of Tuesday afternoon, and I also wanted to check whether you had a chance "
    "to look at the document I sent over on Friday, since there were a couple "
    "of things in the second section that I was not sure about and I would "
    "really like your opinion before we send anything to the client."
)


def sapi_wav(text: str, path: Path) -> Path:
    """Synthesise `text` to `path` using the built-in Windows voice."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{path}'); "
        f"$s.Speak(@'\n{text}\n'@); "
        "$s.Dispose()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
    )
    return path


def load_mono(path: Path, rate: int) -> np.ndarray:
    import soxr

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != rate:
        mono = soxr.resample(mono, sr, rate, quality="VHQ")
    return np.ascontiguousarray(mono, dtype=np.float32)


def main() -> int:
    try:
        return _run()
    finally:
        # Never leave the SAPI reference sitting in the app's voice list: it
        # would show up as a usable profile and the app would happily clone a
        # robot instead of the user.
        import shutil

        from talkbox.enroll import profile_dir
        from talkbox.tts import VoiceCloner

        # save_profile now writes a directory of clips, so removing only the
        # old single .wav would leave the robot voice listed as a real profile.
        shutil.rmtree(profile_dir("_smoketest"), ignore_errors=True)
        for leftover in (
            VoiceCloner.profile_wav("_smoketest"),
            VoiceCloner.profile_cache("_smoketest"),
        ):
            if leftover.exists():
                leftover.unlink()


def _run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="English")
    parser.add_argument("--target", default="Spanish")
    parser.add_argument("--whisper", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--accept-licence",
        action="store_true",
        help="accept the Coqui Public Model License for this run "
        "(setup.ps1 normally records this in config.json)",
    )
    args = parser.parse_args()

    from talkbox.config import Config

    tos = Config.load().coqui_tos_agreed or args.accept_licence
    if not tos:
        print(
            "The XTTS-v2 licence has not been accepted. Run setup.ps1, or pass\n"
            "--accept-licence to agree to the Coqui Public Model License 1.0.0\n"
            "(personal and research use only): https://coqui.ai/cpml"
        )
        return 1

    src = languages.BY_NAME[args.source]
    tgt = languages.BY_NAME[args.target]

    print(f"Talkbox smoke test: {src.name} -> {tgt.name}")
    print("=" * 62)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating stand-in audio with Windows SAPI ...")
    ref_wav = sapi_wav(REFERENCE_TEXT, OUT_DIR / "reference.wav")
    utt_wav = sapi_wav(UTTERANCE_TEXT, OUT_DIR / "utterance.wav")

    # Install the fabricated reference as a real voice profile so the exact
    # code path the app uses at runtime is the one under test.
    from talkbox.enroll import save_profile
    from talkbox.tts import VoiceCloner

    ref_audio, ref_rate = sf.read(ref_wav, dtype="float32")
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)
    save_profile("_smoketest", ref_audio, ref_rate)
    print(f"  reference {ref_audio.size / ref_rate:5.1f}s -> profile '_smoketest'")

    timings: dict[str, float] = {}

    from talkbox.asr import Transcriber
    from talkbox.mt import Translator

    t0 = time.perf_counter()
    print(f"Loading Whisper '{args.whisper}' ...")
    transcriber = Transcriber(args.whisper, args.device, "float16")
    timings["load whisper"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("Loading NLLB-200 ...")
    translator = Translator(device=args.device)
    timings["load nllb"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    print("Loading XTTS-v2 ...")
    cloner = VoiceCloner(
        args.device, progress=lambda m: print(f"  {m}"), tos_agreed=tos
    )
    cloner.load_profile("_smoketest")
    timings["load xtts"] = time.perf_counter() - t0

    speech = load_mono(utt_wav, 16000)
    print(f"\nInput utterance: {speech.size / 16000:.1f}s")
    print("-" * 62)

    # Warm the kernels; the first CUDA call on each model pays for autotuning
    # and would otherwise dominate the numbers below.
    transcriber.transcribe(speech[:16000], src.whisper)
    translator.translate("hello", src.nllb, tgt.nllb)

    t0 = time.perf_counter()
    text, detected, confidence = transcriber.transcribe(speech, src.whisper)
    timings["transcribe"] = time.perf_counter() - t0
    print(f"heard      [{detected}] conf={confidence:.2f} {text!r}")
    if not text:
        print("FAILED: nothing transcribed.")
        return 1

    t0 = time.perf_counter()
    translated = translator.translate(text, src.nllb, tgt.nllb)
    timings["translate"] = time.perf_counter() - t0
    print(f"translated [{tgt.whisper}] {translated!r}")
    if not translated:
        print("FAILED: nothing translated.")
        return 1

    t0 = time.perf_counter()
    chunks: list[np.ndarray] = []
    first_chunk = None
    for chunk in cloner.stream(translated, tgt.xtts):
        if first_chunk is None:
            first_chunk = time.perf_counter() - t0
        chunks.append(chunk)
    timings["synthesise (total)"] = time.perf_counter() - t0
    timings["synthesise (first audio)"] = first_chunk or 0.0

    if not chunks:
        print("FAILED: no audio generated.")
        return 1

    audio = np.concatenate(chunks)
    out_path = OUT_DIR / f"output_{tgt.whisper}.wav"
    sf.write(out_path, audio, 24000, subtype="PCM_16")

    # Long-text pass. Catches the spaCy text-splitting crash and any
    # per-language truncation, neither of which a short sentence reaches.
    print("-" * 62)
    long_translated = translator.translate(LONG_TEXT, src.nllb, tgt.nllb)
    print(f"long input  {len(LONG_TEXT)} chars -> {len(long_translated)} translated")
    t0 = time.perf_counter()
    try:
        long_chunks = list(cloner.stream(long_translated, tgt.xtts))
    except Exception as exc:
        print(f"FAILED on long text: {type(exc).__name__}: {exc}")
        return 1
    if not long_chunks:
        print("FAILED: long text produced no audio.")
        return 1
    long_audio = np.concatenate(long_chunks)
    print(
        f"long synth  {long_audio.size / 24000:.1f}s audio in "
        f"{time.perf_counter() - t0:.1f}s"
    )
    sf.write(OUT_DIR / f"long_{tgt.whisper}.wav", long_audio, 24000, subtype="PCM_16")

    print("-" * 62)
    for label, seconds in timings.items():
        print(f"{label:26} {seconds * 1000:8.0f} ms")

    live = (
        timings["transcribe"]
        + timings["translate"]
        + timings["synthesise (first audio)"]
    )
    print("-" * 62)
    print(f"{'compute to first audio':26} {live * 1000:8.0f} ms")
    print(f"{'+ end-of-phrase pause':26} {320:8.0f} ms")
    print(f"{'= perceived lag':26} {live * 1000 + 320:8.0f} ms")
    print("-" * 62)
    print(f"wrote {out_path}  ({audio.size / 24000:.1f}s)")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
