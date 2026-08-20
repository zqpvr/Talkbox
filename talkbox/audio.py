"""Audio device enumeration, microphone capture, and playback.

Two things make this fiddly on Windows and are worth stating plainly:

PortAudio does not resample. If you ask a device for 16 kHz and its driver only
offers 48 kHz, the stream simply fails to open. So both directions here open at
the device's own default rate and resample in Python with soxr. That costs
well under a millisecond per block and removes a whole class of "works on my
machine" device failures.

Playback is pull-driven from a callback rather than push-driven with
sd.play(), because the TTS produces chunks faster than real time in bursts and
we need them to come out smoothly and gaplessly.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import soxr

_WRAPPED = re.compile(r"^(.*?)\s*\((.+)\)$")
_ENUM_PREFIX = re.compile(r"^\d+-\s*")
_GENERIC_ENDPOINTS = frozenset(
    {
        "speakers",
        "headphones",
        "microphone",
        "headset",
        "headset microphone",
        "headset earphone",
        "line in",
        "digital audio",
        "realtek hd audio mic input",
    }
)

ASR_RATE = 16000
VAD_FRAME = 512  # Silero wants exactly 512 samples at 16 kHz.
TTS_RATE = 24000


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    channels: int
    default_samplerate: float
    hostapi: str
    is_api_default: bool = False

    @property
    def label(self) -> str:
        return f"{self.name}  [{self.hostapi}]"

    @property
    def short(self) -> str:
        """Human-facing name, with Windows' boilerplate stripped.

        "Headset Microphone (3- Astro A20 Voice)" becomes "Astro A20 Voice".
        The parenthetical is only unwrapped when the part before it is a
        generic endpoint word, so "LG ULTRAGEAR (NVIDIA High Definition
        Audio)" and "CABLE Input (VB-Audio Virtual Cable)" keep their real
        identity instead of collapsing to the chipset vendor.
        """
        name = self.name
        match = _WRAPPED.match(name)
        if match and match.group(1).lower() in _GENERIC_ENDPOINTS:
            name = match.group(2)
        return _ENUM_PREFIX.sub("", name).strip() or self.name


def _enumerate(kind: str) -> list[DeviceInfo]:
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    api_key = "default_input_device" if kind == "input" else "default_output_device"
    hostapis = sd.query_hostapis()
    defaults = {api[api_key] for api in hostapis if api.get(api_key, -1) >= 0}
    out: list[DeviceInfo] = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev[key] < 1:
            continue
        out.append(
            DeviceInfo(
                index=idx,
                name=dev["name"],
                channels=dev[key],
                default_samplerate=float(dev["default_samplerate"] or 48000),
                hostapi=hostapis[dev["hostapi"]]["name"],
                is_api_default=idx in defaults,
            )
        )
    return out


# --------------------------------------------------------------------------- #
#  Automatic device selection
# --------------------------------------------------------------------------- #

# Software endpoints that are never what a person means by "my microphone" or
# "my headphones". Steam's pair is listed because it exists on this machine but
# only carries audio during a Remote Play session -- probing it returns silence.
_VIRTUAL = (
    "cable",
    "vb-audio",
    "voicemeeter",
    "steam streaming",
    "virtual",
    "stereo mix",
    "wave out mix",
    "what u hear",
    "loopback",
)

# PortAudio's aggregate pseudo-devices. They work, but they follow the Windows
# default silently, which makes the app's own display of "your microphone"
# wrong the moment the user changes anything.
_AGGREGATE = ("sound mapper", "primary sound", "microsoft soundmapper")

_CABLE_SINKS = ("cable input", "voicemeeter input", "voicemeeter aux input")
_CABLE_SOURCES = ("cable output", "voicemeeter output")

_HEADPHONES = ("headphone", "headset", "earphone", "earbud", "airpod", "buds")

# WASAPI gives the lowest latency and honest device names; MME truncates names
# to 31 characters and adds buffering. WDM-KS often demands exclusive access.
_API_RANK = {
    "Windows WASAPI": 0,
    "Windows DirectSound": 1,
    "MME": 2,
    "Windows WDM-KS": 4,
}


def _has(name: str, needles) -> bool:
    low = name.lower()
    return any(n in low for n in needles)


def is_virtual(dev: DeviceInfo) -> bool:
    return _has(dev.name, _VIRTUAL)


def is_aggregate(dev: DeviceInfo) -> bool:
    return _has(dev.name, _AGGREGATE)


def probe(dev: DeviceInfo, kind: str) -> bool:
    """Open the device briefly to confirm it actually works.

    Enumeration lists endpoints that cannot be opened -- disconnected headsets
    linger, and WASAPI rejects rate or channel combinations that the metadata
    claims to support. Selecting by name alone therefore picks dead devices.
    """
    try:
        cls = sd.InputStream if kind == "input" else sd.OutputStream
        stream = cls(
            device=dev.index,
            samplerate=int(dev.default_samplerate),
            channels=min(dev.channels, 2),
            dtype="float32",
        )
        stream.close()
        return True
    except Exception:
        return False


def _rank(dev: DeviceInfo, prefer_headphones: bool) -> tuple:
    score = _API_RANK.get(dev.hostapi, 3)
    if not dev.is_api_default:
        score += 1
    if prefer_headphones and _has(dev.name, _HEADPHONES):
        score -= 2
    return (score, dev.index)


def autoselect_input(devices: list[DeviceInfo] | None = None) -> DeviceInfo | None:
    """Pick the user's real microphone."""
    devices = devices if devices is not None else input_devices()
    pool = [d for d in devices if not is_virtual(d) and not is_aggregate(d)]
    for dev in sorted(pool, key=lambda d: _rank(d, prefer_headphones=True)):
        if probe(dev, "input"):
            return dev
    # Everything real failed; fall back to whatever will open at all.
    for dev in sorted(devices, key=lambda d: _rank(d, prefer_headphones=True)):
        if probe(dev, "input"):
            return dev
    return None


def autoselect_monitor(devices: list[DeviceInfo] | None = None) -> DeviceInfo | None:
    """Pick the speakers or headphones the user will hear the translation on."""
    devices = devices if devices is not None else output_devices()
    pool = [d for d in devices if not is_virtual(d) and not is_aggregate(d)]
    for dev in sorted(pool, key=lambda d: _rank(d, prefer_headphones=True)):
        if probe(dev, "output"):
            return dev
    return None


def autoselect_call_sink(
    devices: list[DeviceInfo] | None = None,
) -> DeviceInfo | None:
    """Pick the virtual cable input that a call application will hear.

    Returns None when no virtual cable is installed, which is a setup step the
    user has to complete rather than something to silently work around: without
    one there is no way to present audio to Discord or Zoom as a microphone.
    """
    devices = devices if devices is not None else output_devices()
    pool = [d for d in devices if _has(d.name, _CABLE_SINKS)]
    for dev in sorted(pool, key=lambda d: _rank(d, prefer_headphones=False)):
        if probe(dev, "output"):
            return dev
    return None


def is_headphones(dev: DeviceInfo) -> bool:
    """Best-effort guess at whether output is worn rather than in the room.

    Only a guess -- Windows does not report transducer type -- but the naming
    convention is consistent enough to pick a safe default, and the user can
    override it.
    """
    return _has(dev.name, _HEADPHONES)


def is_cable_sink(dev: DeviceInfo) -> bool:
    """True when a call application could actually listen to this endpoint."""
    return _has(dev.name, _CABLE_SINKS)


def input_devices() -> list[DeviceInfo]:
    return _enumerate("input")


def output_devices() -> list[DeviceInfo]:
    return _enumerate("output")


def resolve_device(
    devices: list[DeviceInfo], index: int | None, name: str
) -> DeviceInfo | None:
    """Find a previously chosen device, by name first.

    Names are tried before indices because PortAudio renumbers devices across
    reboots. Returns None when a specific device was asked for and is not
    present -- a headset that is currently unplugged, say. Falling back to
    "device zero" here would silently capture from the wrong microphone, or
    play the translation somewhere the user cannot hear it, and look like it
    had worked. Callers decide whether to re-detect or report the problem.
    """
    if name:
        for dev in devices:
            if dev.name == name:
                return dev
    if index is not None:
        for dev in devices:
            if dev.index == index:
                return dev
    if name or index is not None:
        return None
    return devices[0] if devices else None


class MicCapture:
    """Streams the microphone as 512-sample float32 frames at 16 kHz."""

    def __init__(self, device: DeviceInfo, on_frame, on_level=None):
        self._device = device
        self._on_frame = on_frame
        self._on_level = on_level
        self._rate = int(device.default_samplerate)
        self._channels = min(device.channels, 2)
        self._resampler = (
            soxr.ResampleStream(self._rate, ASR_RATE, 1, dtype="float32")
            if self._rate != ASR_RATE
            else None
        )
        self._tail = np.zeros(0, dtype=np.float32)
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        blocksize = max(256, int(self._rate * 0.032))
        self._stream = sd.InputStream(
            device=self._device.index,
            samplerate=self._rate,
            channels=self._channels,
            dtype="float32",
            blocksize=blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except sd.PortAudioError:
                pass
            self._stream = None

    def _callback(self, indata, _frames, _time, status):  # noqa: ANN001
        if status:
            pass  # Over/underflows are transient; dropping the block is fine.
        mono = indata[:, 0] if self._channels == 1 else indata.mean(axis=1)
        mono = np.ascontiguousarray(mono, dtype=np.float32)

        if self._on_level is not None:
            self._on_level(float(np.sqrt(np.mean(mono**2))))

        if self._resampler is not None:
            mono = self._resampler.resample_chunk(mono)
            if mono.size == 0:
                return
            mono = np.ascontiguousarray(mono.reshape(-1), dtype=np.float32)

        buf = np.concatenate((self._tail, mono)) if self._tail.size else mono
        count = buf.size // VAD_FRAME
        for i in range(count):
            # Copy rather than slice-view: the frame outlives this callback on
            # the VAD queue, and a view would pin the whole parent block.
            self._on_frame(buf[i * VAD_FRAME : (i + 1) * VAD_FRAME].copy())
        self._tail = buf[count * VAD_FRAME :].copy()


class Playback:
    """Gapless output sink. Accepts float32 mono at TTS_RATE."""

    def __init__(self, device: DeviceInfo, src_rate: int = TTS_RATE):
        self._device = device
        self._rate = int(device.default_samplerate)
        self._channels = min(device.channels, 2)
        self._src_rate = int(src_rate)
        self._resampler = (
            soxr.ResampleStream(src_rate, self._rate, 1, dtype="float32")
            if self._rate != src_rate
            else None
        )
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        # Playback can be held back until enough audio is banked. Long
        # sentences generate at roughly real time, so starting the moment the
        # first chunk lands leaves the buffer empty part-way through and the
        # speaker plays silence mid-word.
        self._hold_until = 0.0
        self._holding = False

    def start(self) -> None:
        self._stream = sd.OutputStream(
            device=self._device.index,
            samplerate=self._rate,
            channels=self._channels,
            dtype="float32",
            blocksize=0,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except sd.PortAudioError:
                pass
            self._stream = None

    @property
    def device(self) -> DeviceInfo:
        return self._device

    def write(self, samples: np.ndarray) -> None:
        samples = np.ascontiguousarray(samples.reshape(-1), dtype=np.float32)
        if self._resampler is not None:
            samples = self._resampler.resample_chunk(samples)
            if samples.size == 0:
                return
            samples = np.ascontiguousarray(samples.reshape(-1), dtype=np.float32)
        with self._lock:
            self._buf = np.concatenate((self._buf, samples))

    def hold(self, seconds: float) -> None:
        """Bank this much audio before starting to play."""
        with self._lock:
            self._hold_until = max(0.0, seconds)
            self._holding = self._hold_until > 0.0

    def release(self) -> None:
        """Play whatever is buffered now, however little. Called when
        generation has finished, so a short tail is never left stuck."""
        with self._lock:
            self._holding = False
            self._hold_until = 0.0

    def flush(self) -> None:
        """Drop anything not yet played. Used by the panic/stop button."""
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)
            self._holding = False
            self._hold_until = 0.0
            # soxr's streaming resampler keeps a few samples of history. Left
            # alone, the tail of the sentence the user just cut would be
            # prepended to the next one.
            if self._resampler is not None:
                self._resampler = soxr.ResampleStream(
                    self._src_rate, self._rate, 1, dtype="float32"
                )

    @property
    def queued_seconds(self) -> float:
        with self._lock:
            return self._buf.size / self._rate

    def _callback(self, outdata, frames, _time, status):  # noqa: ANN001
        if status:
            pass
        with self._lock:
            if self._holding:
                if self._buf.size / self._rate >= self._hold_until:
                    self._holding = False
                else:
                    outdata[:] = 0.0
                    return
            take = min(frames, self._buf.size)
            chunk = self._buf[:take]
            self._buf = self._buf[take:]
        if take < frames:
            chunk = np.concatenate((chunk, np.zeros(frames - take, dtype=np.float32)))
        outdata[:] = np.repeat(chunk[:, None], self._channels, axis=1)


class FanoutPlayback:
    """Writes the same audio to several sinks (virtual cable + your headphones).

    `local_only` exists for sounds the user should hear but the call must not:
    error cues, test tones, anything that would be baffling to the person on
    the other end.
    """

    def __init__(self, sinks: list[Playback], local: list[Playback] | None = None):
        self._sinks = sinks
        self._local = local if local is not None else []

    def start(self) -> None:
        for sink in self._sinks:
            sink.start()

    def stop(self) -> None:
        for sink in self._sinks:
            sink.stop()

    def write(self, samples: np.ndarray) -> None:
        for sink in self._sinks:
            sink.write(samples)

    def hold(self, seconds: float) -> None:
        for sink in self._sinks:
            sink.hold(seconds)

    def release(self) -> None:
        for sink in self._sinks:
            sink.release()

    def write_local(self, samples: np.ndarray) -> None:
        """Play only where the user can hear it, never into the call."""
        for sink in self._local or self._sinks[:1]:
            sink.write(samples)

    def flush(self) -> None:
        for sink in self._sinks:
            sink.flush()

    @property
    def queued_seconds(self) -> float:
        return max((s.queued_seconds for s in self._sinks), default=0.0)


def cue_tone(kind: str = "error", rate: int = TTS_RATE) -> np.ndarray:
    """A brief two-tone blip for the monitor sink.

    Deliberately quiet and short. It has to be noticeable while gaming without
    being mistaken for a game sound or drowning out the conversation.
    """
    if kind == "error":
        freqs, seconds, level = (660.0, 440.0), 0.10, 0.16
    else:
        freqs, seconds, level = (520.0, 780.0), 0.09, 0.13
    out = []
    for freq in freqs:
        n = int(rate * seconds)
        t = np.arange(n) / rate
        env = np.minimum(1.0, np.minimum(t * 60.0, (seconds - t) * 60.0))
        out.append((level * np.sin(2 * np.pi * freq * t) * env).astype(np.float32))
    return np.concatenate(out)
