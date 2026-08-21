"""Entry point. `python -m talkbox` launches the GUI; `--doctor` diagnoses."""

from __future__ import annotations

import argparse
import sys

from .runtime import ensure_cuda_dlls


def doctor() -> int:
    """Print an environment report without loading any large model."""
    ok = True
    print("Talkbox environment check")
    print("=" * 60)

    added = ensure_cuda_dlls()
    print(f"CUDA DLL directories registered: {len(added)}")
    for path in added:
        print(f"  {path}")

    try:
        import torch

        print(f"torch                {torch.__version__}")
        print(f"  cuda available     {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  device             {torch.cuda.get_device_name(0)}")
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  vram               {total:.1f} GiB")
        else:
            print("  -> will fall back to CPU, which is too slow for live calls")
            ok = False
    except Exception as exc:
        print(f"torch                MISSING ({exc})")
        ok = False

    for module, label in (
        ("faster_whisper", "faster-whisper"),
        ("transformers", "transformers"),
        ("TTS", "coqui-tts"),
        ("silero_vad", "silero-vad"),
        ("sounddevice", "sounddevice"),
        ("soxr", "soxr"),
        ("PySide6", "PySide6"),
    ):
        try:
            __import__(module)
            print(f"{label:20} ok")
        except Exception as exc:
            print(f"{label:20} MISSING ({exc})")
            ok = False

    print("-" * 60)
    try:
        from . import audio as audio_io

        ins = audio_io.input_devices()
        outs = audio_io.output_devices()
        print(f"input devices  ({len(ins)}):")
        for dev in ins[:12]:
            print(f"  [{dev.index:3}] {dev.name}  ({dev.hostapi})")
        print(f"output devices ({len(outs)}):")
        for dev in outs[:12]:
            print(f"  [{dev.index:3}] {dev.name}  ({dev.hostapi})")
        cable = audio_io.find_virtual_cable(outs)
        print(
            f"virtual cable        {cable.name}"
            if cable
            else "virtual cable        not installed (needed to feed calls)"
        )
    except Exception as exc:
        print(f"audio devices        FAILED ({exc})")
        ok = False

    from .tts import VoiceCloner

    profiles = VoiceCloner.list_profiles()
    print(f"voice profiles       {', '.join(profiles) if profiles else 'none recorded'}")
    if not profiles:
        ok = False

    print("=" * 60)
    print("READY" if ok else "NOT READY - see the lines above")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="talkbox")
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="check GPU, dependencies and audio devices, then exit",
    )
    args = parser.parse_args()

    if args.doctor:
        return doctor()

    ensure_cuda_dlls()
    from .ui import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
