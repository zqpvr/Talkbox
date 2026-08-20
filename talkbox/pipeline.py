"""The orchestrator: microphone in, cloned translated speech out.

Four stages run on their own threads so they pipeline against each other. While
XTTS is synthesising sentence N, Whisper is already transcribing sentence N+1.
On a single GPU the CUDA driver serialises the actual kernels, but the overlap
still hides all the Python, tokenisation and audio-marshalling overhead, which
is a meaningful slice of the wall clock.

    capture (PortAudio callback thread)   -> frame queue
    vad thread: Silero on CPU             -> utterance queue
    recognise thread: Whisper -> NLLB     -> speech queue
    speak thread: XTTS streaming          -> playback ring buffer
                                             -> virtual cable + monitor

The VAD deliberately does not run inside the PortAudio callback. It only costs
about a millisecond per 32 ms frame, but the callback shares a GIL with three
threads that are busy marshalling tensors, and a callback that overruns its
deadline is an audible dropout on the *input* side, which then corrupts the
transcription. A queue hop is cheap insurance.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np

from . import audio as audio_io
from . import languages
from . import quality
from .config import TRANSCRIPT_DIR, Config
from .dictate import Typist


def _noop(*_args) -> None:
    return None


# Short phrases used only to autotune kernels. Each is in its own script so
# that the ja/zh/ko romanisation front ends are exercised during startup rather
# than on the user's first sentence.
_WARMUP_TEXT = {
    "en": "Hello, this is a test.",
    "es": "Hola, esto es una prueba.",
    "fr": "Bonjour, ceci est un test.",
    "de": "Hallo, das ist ein Test.",
    "it": "Ciao, questa e una prova.",
    "pt": "Ola, isto e um teste.",
    "pl": "Czesc, to jest test.",
    "tr": "Merhaba, bu bir test.",
    "ru": "Привет, это тест.",
    "nl": "Hallo, dit is een test.",
    "cs": "Ahoj, toto je test.",
    "ar": "مرحبا، هذا اختبار.",
    "zh": "你好，这是一个测试。",
    "ja": "こんにちは、これはテストです。",
    "hu": "Szia, ez egy teszt.",
    "ko": "안녕하세요, 이것은 테스트입니다.",
    "hi": "नमस्ते, यह एक परीक्षण है।",
}


@dataclass
class Events:
    on_status: Callable[[str], None] = _noop
    on_ready: Callable[[], None] = _noop
    on_source: Callable[[str], None] = _noop
    on_translation: Callable[[str], None] = _noop
    on_latency: Callable[[float], None] = _noop
    on_level: Callable[[float], None] = _noop
    on_speech: Callable[[bool], None] = _noop
    on_error: Callable[[str], None] = _noop
    on_busy: Callable[[int, int], None] = _noop
    on_unclear: Callable[[str], None] = _noop
    on_talkover: Callable[[], None] = _noop
    # text, and whether it was actually typed somewhere. Not typed means our
    # own window had focus, so the user is shown the sentence instead.
    on_dictation: Callable[[str, bool], None] = _noop


@dataclass
class _Job:
    audio: np.ndarray
    captured_at: float
    speech_frames: int = 0
    speculative: bool = False


@dataclass
class _Guess:
    """A transcription and translation produced before the pause finished."""

    speech_frames: int
    direction: tuple[str, str, bool]
    text: str
    detected: str
    translated: str
    target: languages.Language


@dataclass
class _Utterance:
    text: str
    lang: languages.Language
    captured_at: float
    warmup: bool = False


class Pipeline:
    def __init__(self, cfg: Config, events: Events | None = None):
        self.cfg = cfg
        self.ev = events or Events()

        self.transcriber = None
        self.translator = None
        self.cloner = None
        # What the status line should say once everything is up. The speak
        # thread prepares the voice after start() has already reported
        # Live, so it has to put this back afterwards.
        self._live_status = ""
        # Set when the voice model could not be loaded, so Talk mode can
        # explain why rather than failing silently at the first sentence.
        self.voice_error: str | None = None
        self._typist = Typist()
        self._loaded = False

        self._lock = threading.RLock()
        self._running = False
        self._threads: list[threading.Thread] = []
        # ~8 seconds of 32 ms frames; if the VAD ever falls this far behind
        # something is badly wrong and dropping is better than unbounded growth.
        self._frame_q: queue.Queue = queue.Queue(maxsize=256)
        self._utt_q: queue.Queue[_Job | None] = queue.Queue(maxsize=8)
        self._say_q: queue.Queue[_Utterance | None] = queue.Queue(maxsize=8)

        self._mic: audio_io.MicCapture | None = None
        self._out: audio_io.FanoutPlayback | None = None
        self._segmenter = None
        self._transcript = None

        self._auto_direction = False
        self._guard_until = 0.0
        # Incremented on every start/stop. A synthesis that overran the join
        # timeout would otherwise still be holding a reference to the pipeline
        # and could write stale audio into the *next* session's output device.
        self._session = 0
        # Result of speculative recognition, waiting to be claimed by the real
        # utterance when the pause is confirmed.
        self._guess: _Guess | None = None
        self._guess_lock = threading.Lock()
        self.guesses_used = 0
        self.guesses_wasted = 0

    # -- model loading --------------------------------------------------------

    def load_models(self) -> None:
        """Blocking. Call from a worker thread; progress arrives via on_status."""
        if self._loaded:
            self.ev.on_ready()
            return
        try:
            import torch

            device = self.cfg.device
            if device == "cuda" and not torch.cuda.is_available():
                self.ev.on_status("No CUDA GPU. Running on the CPU.")
                device = "cpu"
                self.cfg.device = "cpu"
                self.cfg.compute_type = "int8"

            from .asr import Transcriber
            from .mt import Translator
            from .tts import VoiceCloner

            preset = quality.speech(self.cfg)
            whisper_model = preset.whisper_model
            if device == "cpu" and whisper_model not in ("tiny", "base"):
                # Measured on an i7-14700F, 20 cores, for a 2.9 s sentence:
                # tiny 234 ms, small 1944 ms. Small is unusable for dictation
                # and none of them are fast enough for a call, so on the CPU
                # the recogniser is capped rather than left at whatever the
                # quality dropdown says.
                self.ev.on_status(
                    f"On the CPU, using Whisper 'base' instead of "
                    f"'{whisper_model}'."
                )
                whisper_model = "base"
            self.ev.on_status(
                f"Loading Whisper '{whisper_model}' ({preset.name})..."
            )
            self.transcriber = Transcriber(
                whisper_model, device, self.cfg.compute_type
            )

            short = preset.mt_model.rsplit("-", 1)[-1]
            self.ev.on_status(f"Loading NLLB-200 translator ({short})...")
            self.translator = Translator(
                preset.mt_model, device, progress=self.ev.on_status
            )
            self.ev.on_status(f"Translator ready ({self.translator.backend}).")

            # The voice model is the largest of the three and the only one
            # dictation never touches, so a failure here is not fatal. Report
            # it and carry on: recognition and translation are enough to
            # dictate with, and the Talk mode will say so when it is used.
            try:
                self.cloner = VoiceCloner(
                    device,
                    progress=self.ev.on_status,
                    tos_agreed=self.cfg.coqui_tos_agreed,
                )
            except Exception as exc:
                self.cloner = None
                self.voice_error = str(exc)
                self.ev.on_error(
                    f"Voice model unavailable: {exc} Dictation still works."
                )

            self.ev.on_status("Warming up CUDA kernels...")
            self._warm_recognition()

            self._loaded = True
            self.ev.on_status("Models ready.")
            self.ev.on_ready()
        except Exception as exc:  # surfaced in the UI rather than a traceback
            self.ev.on_error(f"Model load failed: {exc}")

    def _warm_recognition(self) -> None:
        """Force CUDA to autotune the Whisper and NLLB kernel shapes now.

        Cold, the first utterance costs about ten seconds end to end: nearly
        four in Whisper, another four in NLLB and close to three before XTTS
        yields audio. Steady state is under two. Paying that during startup
        instead of on the first thing the user says is the difference between
        a usable call opening and a dead thirty seconds.
        """
        try:
            import numpy as np

            for seconds in (2.0, 6.0):  # Two shapes; CT2 tunes per length.
                silence = np.zeros(int(16000 * seconds), dtype=np.float32)
                self.transcriber.transcribe(silence, "en")  # 3-tuple, discarded
            for probe in ("Hello there.", "I would like to talk about the schedule."):
                self.translator.translate(probe, "eng_Latn", "spa_Latn")
        except Exception as exc:
            self.ev.on_status(f"Warm-up skipped: {exc}")

    def _warm_synthesis(self) -> None:
        """Queue a throwaway synthesis so XTTS is hot before the first line.

        It runs in the real target language rather than a fixed phrase, because
        Japanese, Chinese and Korean are routed through separate romanisation
        and segmentation front ends. Warming the wrong one would leave the
        expensive path cold exactly when it is first needed.
        """
        lang = languages.get(self.cfg.target_lang)
        try:
            self._say_q.put_nowait(
                _Utterance(
                    _WARMUP_TEXT.get(lang.whisper, "Hello, this is a test."),
                    lang,
                    time.monotonic(),
                    warmup=True,
                )
            )
        except queue.Full:
            pass

    @property
    def loaded(self) -> bool:
        return self._loaded

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            if not self._loaded:
                raise RuntimeError("Models are not loaded yet.")

            # Only check the recording exists here. Deriving the speaker
            # latents can take several seconds at high cloning quality, and
            # start() is called straight from the GUI thread, so the actual
            # work is done by the speak thread before its first utterance.
            if self.cfg.mode != "dictate" and self.cloner is None:
                raise RuntimeError(
                    "The voice model did not load, so nothing can be spoken. "
                    + (self.voice_error or "")
                    + " Switch to typing it out instead, on the Talk tab."
                )
            if (
                self.cfg.mode != "dictate"
                and self.cfg.voice_mode != "builtin"
                and not self.cloner.profile_clips(self.cfg.speaker_profile)
            ):
                raise FileNotFoundError(
                    f"No voice profile named '{self.cfg.speaker_profile}'. "
                    "Record one, or switch to a built-in voice."
                )

            ins = audio_io.input_devices()
            outs = audio_io.output_devices()
            mic_dev = audio_io.resolve_device(
                ins, self.cfg.input_device, self.cfg.input_device_name
            )
            out_dev = audio_io.resolve_device(
                outs, self.cfg.output_device, self.cfg.output_device_name
            )

            # A remembered device can simply be gone -- an unplugged headset,
            # or a cable uninstalled since last run. Re-detect rather than
            # refusing to start or, worse, grabbing an arbitrary endpoint.
            if mic_dev is None:
                mic_dev = audio_io.autoselect_input(ins)
                if mic_dev is not None:
                    self.ev.on_status(
                        f"Saved microphone is gone; using {mic_dev.short}."
                    )
            if out_dev is None:
                out_dev = audio_io.autoselect_call_sink(
                    outs
                ) or audio_io.autoselect_monitor(outs)
                if out_dev is not None:
                    self.ev.on_status(
                        f"Saved output is gone; using {out_dev.short}."
                    )
            if mic_dev is None:
                raise RuntimeError("No working microphone could be opened.")
            # Dictation produces no audio, so it must not be blocked by a
            # missing virtual cable. That is the whole reason it works on a
            # machine where the driver was never installed.
            dictating = self.cfg.mode == "dictate"
            if out_dev is None and not dictating:
                raise RuntimeError("No working output device could be opened.")

            sinks = [audio_io.Playback(out_dev)] if out_dev is not None else []
            if self.cfg.monitor_enabled and not dictating:
                mon = audio_io.resolve_device(
                    outs, self.cfg.monitor_device, self.cfg.monitor_device_name
                )
                if mon is not None and mon.index != out_dev.index:
                    sinks.append(audio_io.Playback(mon))
            # Only the monitor is 'local'; cue tones must never reach the
            # cable, or the person on the other end hears them too.
            if sinks and not dictating:
                self._out = audio_io.FanoutPlayback(
                    sinks, local=sinks[1:] if len(sinks) > 1 else []
                )
                self._out.start()
            else:
                self._out = None

            # The guard exists to stop speakers feeding the synthesised voice
            # back into the microphone. On headphones that cannot happen, and
            # keeping it on needlessly blocks the user from talking over the
            # output. Decide from the device unless the user set it manually.
            # Nothing is played while dictating, so the microphone never has
            # to be gated against it.
            if dictating:
                self.cfg.duplex_guard = False
            elif self.cfg.duplex_guard_auto:
                # Whatever the user actually hears. With no monitor sink that
                # is the main output; a virtual cable is inaudible in the room,
                # so it can never feed back.
                audible = sinks[-1].device if len(sinks) > 1 else out_dev
                headphones = audio_io.is_headphones(audible) or audio_io.is_cable_sink(
                    audible
                )
                self.cfg.duplex_guard = not headphones
                self.ev.on_status(
                    "Headphones detected - you can talk over the translation."
                    if headphones
                    else "Speakers detected - mic muted while it speaks."
                )

            from .vad import Segmenter

            self._segmenter = Segmenter(
                on_utterance=self._enqueue,
                on_provisional=(
                    self._enqueue_guess if self.cfg.speculative_recognition else None
                ),
                speculate_after_ms=self.cfg.speculate_after_ms,
                speculate_interval_ms=self.cfg.speculate_interval_ms,
                early_silence_ms=self.cfg.early_silence_ms,
                on_speech_state=self.ev.on_speech,
                threshold=self.cfg.vad_threshold,
                min_silence_ms=self.cfg.min_silence_ms,
                preroll_ms=self.cfg.preroll_ms,
                min_speech_ms=self.cfg.min_speech_ms,
                max_utterance_s=self.cfg.max_utterance_s,
                hold_silence_ms=self.cfg.hold_silence_ms,
                lookback_ms=self.cfg.lookback_ms,
                pause_window_ms=self.cfg.pause_window_ms,
            )

            if self.cfg.save_transcript:
                TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                self._transcript = (TRANSCRIPT_DIR / f"{stamp}.txt").open(
                    "a", encoding="utf-8"
                )

            # Clear anything left from a previous session. A worker that died
            # early -- a voice profile that would not load, say -- never
            # consumed its stop sentinel, and that stale None would end the
            # next session's thread the moment it started, silently producing
            # no audio at all.
            for q in (self._frame_q, self._utt_q, self._say_q):
                self._drain(q)
            # A new session starts a new paragraph; no space is owed from
            # whatever was dictated last time.
            self._typist.reset()
            with self._guess_lock:
                self._guess = None

            self._running = True
            self._session += 1
            self._threads = [
                threading.Thread(target=self._vad_loop, daemon=True),
                threading.Thread(target=self._recognise_loop, daemon=True),
                threading.Thread(target=self._speak_loop, daemon=True),
            ]
            for t in self._threads:
                t.start()

            self._mic = audio_io.MicCapture(
                mic_dev, self._push_frame, on_level=self.ev.on_level
            )
            self._mic.start()
            # Runs on the speak thread, so it does not stall the caller; the
            # mic gate ignores it because nothing reaches the output device.
            self._warm_synthesis()
            self._live_status = (
                f"Live: {mic_dev.name}, typing into whatever window is in front"
                if dictating
                else f"Live: {mic_dev.name} -> {out_dev.name}"
            )
            self.ev.on_status(self._live_status)

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False

            if self._mic:
                self._mic.stop()
                self._mic = None

            with self._guess_lock:
                self._guess = None
            self._drain(self._frame_q)
            self._drain(self._utt_q)
            self._drain(self._say_q)
            self._frame_q.put(None)
            self._utt_q.put(None)
            self._say_q.put(None)

        for t in self._threads:
            t.join(timeout=5.0)
        # Anything that outlived the join -- a synthesis stuck in a slow first
        # chunk, say -- sees a changed session id and drops its output instead
        # of playing it into whatever comes next.
        self._session += 1
        self._threads = []

        if self._out:
            self._out.flush()
            self._out.stop()
            self._out = None
        if self._transcript:
            self._transcript.close()
            self._transcript = None
        self._segmenter = None
        self.ev.on_status("Stopped.")

    @property
    def running(self) -> bool:
        return self._running

    def flush(self) -> None:
        """Panic button: drop everything queued and stop mid-sentence."""
        with self._guess_lock:
            self._guess = None
        self._drain(self._frame_q)
        self._drain(self._utt_q)
        self._drain(self._say_q)
        if self._out:
            self._out.flush()
        if self._segmenter:
            self._segmenter.reset()

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    # -- direction ------------------------------------------------------------

    def set_direction(self, source: str, target: str, auto: bool) -> None:
        with self._lock:
            self.cfg.source_lang = source
            self.cfg.target_lang = target
            self._auto_direction = auto

    def _direction(self) -> tuple[languages.Language, languages.Language, bool]:
        with self._lock:
            return (
                languages.get(self.cfg.source_lang),
                languages.get(self.cfg.target_lang),
                self._auto_direction,
            )

    # -- stages ---------------------------------------------------------------

    def _push_frame(self, frame: np.ndarray) -> None:
        """Called on the PortAudio thread. Must never block."""
        try:
            self._frame_q.put_nowait(frame)
        except queue.Full:
            pass

    def _muted_by_output(self) -> bool:
        """True while our own translated speech is audible.

        Without this, anyone monitoring through speakers rather than a headset
        creates a feedback loop: the mic hears the synthesised Spanish, Whisper
        transcribes it, and it gets translated straight back into English in
        the same voice, forever.
        """
        if not self.cfg.duplex_guard or self._out is None:
            return False
        if self._out.queued_seconds > 0.02:
            return True
        return time.monotonic() < self._guard_until

    def _vad_loop(self) -> None:
        gated = False
        warned = False
        loud_frames = 0
        while self._running:
            frame = self._frame_q.get()
            if frame is None:
                break
            try:
                if self._muted_by_output():
                    # Drop the frame, and clear any half-built utterance so the
                    # first thing after the gate lifts is a clean onset.
                    if not gated:
                        self._segmenter.reset()
                        gated = True
                        warned = False
                        loud_frames = 0
                    # Losing speech silently is the worst part of the gate, so
                    # at least say so. Energy is enough here: the VAD is not
                    # run while gated, and anything this loud at the mic is
                    # either the user or a room loud enough to be a problem.
                    if not warned:
                        if float(np.sqrt(np.mean(frame**2))) > self.cfg.talkover_rms:
                            loud_frames += 1
                            if loud_frames >= 8:  # about 250 ms
                                warned = True
                                self.ev.on_talkover()
                        else:
                            loud_frames = 0
                    continue
                gated = False
                loud_frames = 0
                self._segmenter.push(frame)
            except Exception as exc:
                self.ev.on_error(f"VAD failed: {exc}")

    def _enqueue_guess(self, samples: np.ndarray, speech_frames: int) -> None:
        """Queue a speculative pass. Dropped silently if the queue is busy;
        it is only ever an optimisation."""
        try:
            self._utt_q.put_nowait(
                _Job(samples, time.monotonic(), speech_frames, speculative=True)
            )
        except queue.Full:
            pass

    def _enqueue(self, samples: np.ndarray, speech_frames: int = 0) -> None:
        job = _Job(samples, time.monotonic(), speech_frames)
        try:
            self._utt_q.put_nowait(job)
        except queue.Full:
            # Backed up. Drop the oldest so the conversation keeps moving
            # rather than drifting further and further behind.
            try:
                self._utt_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._utt_q.put_nowait(job)
            except queue.Full:
                pass
        self.ev.on_busy(self._utt_q.qsize(), self._say_q.qsize())

    def _recognise_loop(self) -> None:
        while self._running:
            job = self._utt_q.get()
            if job is None:
                break
            try:
                src, tgt, auto = self._direction()
                direction = (src.whisper, tgt.whisper, auto)

                claimed = None
                if not job.speculative:
                    claimed = self._claim_guess(job.speech_frames, direction)

                if claimed is not None:
                    # Already transcribed and translated during the pause.
                    text, detected = claimed.text, claimed.detected
                    translated, tgt = claimed.translated, claimed.target
                    confidence = 1.0
                else:
                    text, detected, confidence = self.transcriber.transcribe(
                        job.audio, language=None if auto else src.whisper
                    )
                    if not text or confidence < self.cfg.min_confidence:
                        if job.speculative:
                            continue
                        if text:
                            # Show it, never speak it. A misheard sentence in
                            # the user's own voice on a live call is worse than
                            # a gap they can repeat.
                            self.ev.on_unclear(text)
                            self._cue("error")
                        continue

                    # Decide about ending the pause early here, before
                    # translating. Translation adds around 160 ms, and the
                    # pause confirms at 400 ms, so granting permission after it
                    # arrived too late to ever be used.
                    if job.speculative and self.cfg.early_commit:
                        segmenter = self._segmenter
                        if segmenter is not None:
                            if self._looks_finished(text, confidence):
                                segmenter.allow_early_commit(job.speech_frames)
                            else:
                                # Mid-clause. Tell the segmenter to keep the
                                # pause open, so a breath partway through a
                                # long sentence does not end it.
                                segmenter.hold_commit(job.speech_frames)

                    if auto and detected == tgt.whisper:
                        src, tgt = tgt, src
                    if self.cfg.mode == "dictate" and not self.cfg.dictate_translate:
                        # Plain dictation: write down what was said, in the
                        # language it was said in.
                        translated, tgt = text, src
                    else:
                        translated = self.translator.translate(
                            text, src.nllb, tgt.nllb
                        )
                        if not translated:
                            continue

                if job.speculative:
                    # Park it. The real utterance claims it moments later.
                    self._store_guess(
                        _Guess(
                            job.speech_frames, direction, text, detected, translated, tgt
                        )
                    )
                    continue

                self.ev.on_source(text)
                self.ev.on_translation(translated)
                self._log(src, text, tgt, translated)

                if self.cfg.mode == "dictate":
                    typed = self._typist.type(translated)
                    self.ev.on_dictation(translated, typed)
                    continue

                try:
                    self._say_q.put_nowait(
                        _Utterance(translated, tgt, job.captured_at)
                    )
                except queue.Full:
                    self.ev.on_status("Speech queue full - dropped a line.")
            except Exception as exc:
                self.ev.on_error(f"Recognition failed: {exc}")
            finally:
                self.ev.on_busy(self._utt_q.qsize(), self._say_q.qsize())

    def _speak_loop(self) -> None:
        session = self._session
        if self.cfg.mode == "dictate":
            # Nothing is ever queued for it, but the thread still has to run
            # so stop() finds something to hand its sentinel to.
            while self._running and session == self._session:
                if self._say_q.get() is None:
                    break
            return
        # If the voice will not load there is nothing to synthesise, but the
        # thread stays alive to drain the queue and consume its own stop
        # sentinel. Exiting here used to strand that sentinel for the next run.
        voice_ok = self._prepare_voice()
        # Clear the "Loading voice..." line. It is emitted from this thread
        # after start() has already reported Live, so without this it is the
        # last thing written and sits there for the whole session.
        if voice_ok and self._live_status and session == self._session:
            self.ev.on_status(self._live_status)
        while self._running and session == self._session:
            item = self._say_q.get()
            if item is None:
                break
            if not voice_ok:
                self.ev.on_busy(self._utt_q.qsize(), self._say_q.qsize())
                continue
            try:
                first = True
                cap = self._runaway_cap(item.text, item.lang)
                emitted = 0
                out0 = self._out
                if out0 is not None and not item.warmup:
                    out0.hold(self._prebuffer_for(item.text, item.lang))
                preset = quality.speech(self.cfg)
                temperature, top_k, top_p = quality.voice(self.cfg)
                for chunk in self.cloner.stream(
                    item.text,
                    item.lang.xtts,
                    chunk_size=preset.tts_chunk_size,
                    temperature=temperature,
                    speed=self.cfg.tts_speed,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=preset.tts_repetition_penalty,
                ):
                    if not self._running or session != self._session:
                        break
                    if first and not item.warmup:
                        self.ev.on_latency(time.monotonic() - item.captured_at)
                        first = False

                    if item.warmup:
                        # Generated purely to autotune the kernels. Two chunks
                        # is enough to cover the expensive first pass.
                        emitted += chunk.size
                        if emitted / audio_io.TTS_RATE > 0.5:
                            break
                        continue

                    emitted += chunk.size
                    if emitted / audio_io.TTS_RATE > cap:
                        self.ev.on_status(
                            f"Synthesis ran long ({cap:.0f}s cap) - truncated."
                        )
                        break

                    out = self._out
                    if out is not None and session == self._session:
                        out.write(chunk)
                        # Hold the mic gate a little past the last sample so
                        # the room's acoustic tail is not treated as speech.
                        self._guard_until = (
                            time.monotonic() + self.cfg.guard_hangover_ms / 1000.0
                        )
            except Exception as exc:
                self.ev.on_error(f"Synthesis failed: {exc}")
            finally:
                # Never leave audio stuck behind the hold. Guarded because an
                # exception raised in a finally block would propagate out of
                # the loop and kill this thread for the rest of the session.
                try:
                    out_end = self._out
                    if out_end is not None:
                        out_end.release()
                except Exception:
                    pass
                self.ev.on_busy(self._utt_q.qsize(), self._say_q.qsize())

    def _prepare_voice(self) -> bool:
        """Derive speaker latents before the first utterance is handled.

        At high cloning quality this reads up to 30 seconds of reference audio
        through the GPT encoder, which is far too slow to sit on the GUI thread
        where start() is called from. A built-in speaker skips that entirely:
        its latents are already in the checkpoint, so loading one is a tensor
        move and measured at 0 ms against 101 ms for a recorded profile.
        """
        if self.cfg.voice_mode == "builtin":
            try:
                self.ev.on_status("Loading voice...")
                self.cloner.load_builtin(self.cfg.builtin_voice)
                return True
            except Exception as exc:
                self.ev.on_error(f"Could not load built-in voice: {exc}")
                return False

        clone = quality.clone(self.cfg)
        try:
            self.ev.on_status("Preparing your voice...")
            self.cloner.load_profile(
                self.cfg.speaker_profile,
                gpt_cond_len=clone.gpt_cond_len,
                max_ref_length=clone.max_ref_length,
            )
            return True
        except Exception as exc:
            # A recording that has gone missing or will not decode should not
            # leave the user with a pipeline that runs but never speaks.
            self.ev.on_error(
                f"Could not load voice profile: {exc}. "
                "Falling back to a built-in voice."
            )
            try:
                self.cloner.load_builtin(self.cfg.builtin_voice)
                return True
            except Exception:
                return False

    # Characters per second of synthesised speech, measured per language.
    # A single global figure does not work: Japanese and Chinese pack far more
    # meaning into each character, and measured about 7 characters per second
    # against 13 for Spanish and 16 for Russian. Using 15 everywhere put
    # Japanese at 86% of its cap on a normal sentence, one unlucky run away
    # from having real speech cut off.
    _CHARS_PER_SECOND = {"ja": 6.0, "zh": 6.0, "ko": 6.0, "ar": 10.0, "hi": 10.0}
    _DEFAULT_CHARS_PER_SECOND = 12.0

    def _prebuffer_for(self, text: str, lang: languages.Language) -> float:
        """How much audio to bank before playback starts.

        Short sentences need none: measured worst-case buffer stayed positive.
        Long ones generate at roughly real time, so without a head start the
        buffer empties part-way through and the speaker plays a gap. Scaling
        with the expected duration costs nothing on the sentences that make up
        normal conversation.
        """
        expected = len(text) / self._CHARS_PER_SECOND.get(
            lang.whisper, self._DEFAULT_CHARS_PER_SECOND
        )
        if expected <= self.cfg.prebuffer_below_s:
            return 0.0
        return min(
            self.cfg.prebuffer_max_s, expected * self.cfg.prebuffer_fraction
        )

    def _runaway_cap(self, text: str, lang: languages.Language | None = None) -> float:
        """Upper bound on synthesised seconds for a line of text.

        Only a backstop against XTTS looping instead of stopping. It should
        never fire on real speech.
        """
        code = lang.whisper if lang is not None else "en"
        rate = self._CHARS_PER_SECOND.get(code, self._DEFAULT_CHARS_PER_SECOND)
        expected = len(text) / rate
        return max(self.cfg.tts_min_cap_s, expected * self.cfg.tts_runaway_factor)

    # Sentence-final punctuation across the scripts in use, including the
    # full-width CJK forms and the Devanagari danda.
    _TERMINAL = ".!?\u3002\uff01\uff1f\u06d4\u0964"

    def _looks_finished(self, text: str, confidence: float) -> bool:
        """Is this guess a whole sentence, not a clause cut short?

        Measured on partial cuts of a three-sentence passage: trailing
        punctuation identified every real sentence end and rejected every
        mid-clause cut but one, where Whisper invented an ending. That one
        scored 0.77 confidence, so requiring high confidence as well filters
        it out.
        """
        text = text.rstrip()
        if not text or text[-1] not in self._TERMINAL:
            return False
        return confidence >= self.cfg.early_commit_confidence

    def _store_guess(self, guess: _Guess) -> None:
        with self._guess_lock:
            if self._guess is not None:
                self.guesses_wasted += 1
            self._guess = guess

    def _claim_guess(
        self, speech_frames: int, direction: tuple[str, str, bool]
    ) -> _Guess | None:
        """Take a stored guess if it still applies.

        It applies when no further speech arrived after it was made and the
        translation direction has not changed. Matching on the speech-frame
        count is what makes this safe: any new speech increments it, so a stale
        guess can never be mistaken for a current one.
        """
        with self._guess_lock:
            guess = self._guess
            self._guess = None
        if guess is None:
            return None
        if guess.speech_frames != speech_frames or guess.direction != direction:
            self.guesses_wasted += 1
            return None
        self.guesses_used += 1
        return guess

    def _cue(self, kind: str = "error") -> None:
        """Tell the user something happened without making them alt-tab.

        Routed to the monitor only. If there is no separate monitor -- the user
        is listening on the same device the call hears -- it is skipped rather
        than played into the call.
        """
        out = self._out
        if out is None or not getattr(out, "_local", None):
            return
        try:
            out.write_local(audio_io.cue_tone(kind))
        except Exception:
            pass

    def _log(
        self,
        src: languages.Language,
        text: str,
        tgt: languages.Language,
        translated: str,
    ) -> None:
        if not self._transcript:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        self._transcript.write(
            f"[{stamp}] {src.name}: {text}\n[{stamp}] {tgt.name}: {translated}\n\n"
        )
        self._transcript.flush()
