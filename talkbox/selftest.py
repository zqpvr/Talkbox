"""End-to-end verification of the audio path a call actually depends on.

The Setup panel can tell you a virtual cable exists. It cannot tell you the
cable is unmuted, that Windows has not routed it somewhere unexpected, or that
the sample rates line up. This plays real synthesised speech into the cable and
records it back off the cable's capture side, which is exactly what the call
application will be listening to.

Failing here before a call is worth a great deal more than discovering it while
someone waits for you to say something.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from . import audio as audio_io


@dataclass
class Result:
    ok: bool
    headline: str
    detail: str


def _find_cable_source() -> audio_io.DeviceInfo | None:
    """The capture endpoint a call app would be pointed at."""
    candidates = [
        d
        for d in audio_io.input_devices()
        if audio_io._has(d.name, audio_io._CABLE_SOURCES)
    ]
    for dev in sorted(candidates, key=lambda d: audio_io._rank(d, False)):
        if audio_io.probe(dev, "input"):
            return dev
    return None


def _energy(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal**2))) if signal.size else 0.0


def run(
    cloner,
    sink: audio_io.DeviceInfo | None,
    language: str = "en",
    phrase: str = "Testing, one two three. If you can hear this, the setup works.",
    progress=None,
) -> Result:
    """Synthesise, play into the cable, capture from it, and compare."""

    def say(message: str) -> None:
        if progress:
            progress(message)

    if sink is None:
        return Result(False, "No output selected", "Pick an output device first.")

    if not audio_io.is_cable_sink(sink):
        return Result(
            False,
            "Not a virtual cable",
            f"{sink.short} is a normal output. A call application cannot "
            "listen to it, so there is nothing to verify. Install VB-Cable and "
            "select CABLE Input.",
        )

    source = _find_cable_source()
    if source is None:
        return Result(
            False,
            "Cable capture side missing",
            "Found CABLE Input but no matching CABLE Output to record from. "
            "The driver may be half-installed; reinstall VB-Cable and reboot.",
        )

    say("Synthesising a test phrase in your voice...")
    try:
        speech = np.concatenate(list(cloner.stream(phrase, language, chunk_size=20)))
    except Exception as exc:
        return Result(False, "Could not synthesise", str(exc))
    if speech.size == 0:
        return Result(False, "Could not synthesise", "No audio was produced.")

    say(f"Playing into {sink.short} and listening on {source.short}...")
    captured: list[np.ndarray] = []
    rate = int(source.default_samplerate)
    channels = min(source.channels, 2)

    def on_capture(indata, _frames, _time, _status):  # noqa: ANN001
        block = indata[:, 0] if channels == 1 else indata.mean(axis=1)
        captured.append(np.ascontiguousarray(block, dtype=np.float32))

    playback = audio_io.Playback(sink)
    try:
        stream = sd.InputStream(
            device=source.index,
            samplerate=rate,
            channels=channels,
            dtype="float32",
            callback=on_capture,
        )
    except Exception as exc:
        return Result(
            False, "Could not open the cable for recording", f"{type(exc).__name__}: {exc}"
        )

    try:
        stream.start()
        playback.start()
        time.sleep(0.25)  # Let both streams settle before measuring.
        baseline = np.concatenate(captured) if captured else np.zeros(1, np.float32)
        captured.clear()

        playback.write(speech)
        duration = speech.size / audio_io.TTS_RATE
        deadline = time.monotonic() + duration + 1.5
        while time.monotonic() < deadline:
            if playback.queued_seconds <= 0.01 and time.monotonic() > deadline - 1.2:
                break
            time.sleep(0.05)
        time.sleep(0.2)
        heard = np.concatenate(captured) if captured else np.zeros(1, np.float32)
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        playback.stop()

    quiet = _energy(baseline)
    loud = _energy(heard)
    expected = _energy(speech)
    say("")

    if loud < 0.002:
        return Result(
            False,
            "The cable carried no audio",
            f"Played {duration:.1f}s into {sink.short} but recorded silence on "
            f"{source.short}. Check that neither endpoint is muted in the "
            "Windows volume mixer, and that CABLE Output is not disabled.",
        )
    if loud < max(quiet * 3.0, 0.004):
        return Result(
            False,
            "Signal too weak to be sure",
            f"Recorded {loud:.4f} RMS against a {quiet:.4f} noise floor. Raise "
            f"the level of {sink.short} in the Windows volume mixer.",
        )

    ratio = loud / expected if expected else 0.0
    detail = (
        f"Played {duration:.1f}s of your cloned voice into {sink.short} and "
        f"recorded it back on {source.short} at {loud:.3f} RMS "
        f"({ratio * 100:.0f}% of the synthesised level). "
        f"Point your call application's microphone at {source.name}."
    )
    return Result(True, "Setup verified — a call would hear this", detail)


def run_async(cloner, sink, language, callback, progress=None) -> threading.Thread:
    """Run the test off the GUI thread; `callback` receives a Result."""

    def worker() -> None:
        try:
            result = run(cloner, sink, language, progress=progress)
        except Exception as exc:  # never let a probe take the app down
            result = Result(False, "Test failed", f"{type(exc).__name__}: {exc}")
        callback(result)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread
