"""Voice activity detection and utterance segmentation.

Silero VAD runs on the CPU. It is a 2 MB model, and keeping it off the GPU
means endpoint detection stays on time even when Whisper and XTTS are busy.

Where the cuts land matters more than how fast they happen. How long a pause
has to run before it counts as the end of a sentence depends on what the
speculative transcript looks like:

    normally            min_silence_ms (400 ms)
    guess ends in .!?   early_silence_ms (240 ms), because it reads as finished
    guess is mid-clause hold_silence_ms (700 ms), because it clearly is not
    at max_utterance_s  no pause has come at all, so cut at the quietest point
                        in the last few seconds rather than at the current frame

An earlier version shortened the pause to 180 ms for anything past eight
seconds, on the theory that a long utterance should not also wait a long time
to end. That cut long sentences in half: a breath between clauses is commonly
around 200 ms, so any sentence that ran past eight seconds ended at the next
time its speaker inhaled. The rule is now the other way round. Length alone
proves nothing, and past that point a transcript still sitting mid-clause is
treated as evidence the speaker has not finished, so the pause is held open
longer rather than cut shorter.

The last rule matters too. Cutting at an arbitrary timer lands mid-word, and
each fragment is then translated on its own, which is where bad output comes
from. Searching back for the quietest run finds a real gap between words most
of the time.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable

import numpy as np
import torch

from .audio import ASR_RATE, VAD_FRAME

FRAME_MS = VAD_FRAME * 1000 // ASR_RATE  # 32 ms
HYSTERESIS = 0.15

_MODEL = None
_MODEL_LOCK = threading.Lock()


def _shared_model():
    """Load Silero once per process.

    A Segmenter is built on every Start. Reloading a TorchScript module each
    time adds a pause before the first frame is looked at.
    """
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            from silero_vad import load_silero_vad

            _MODEL = load_silero_vad()
        return _MODEL


class Segmenter:
    """Turns a stream of 512-sample frames into complete utterances."""

    def __init__(
        self,
        on_utterance: Callable[[np.ndarray, int], None],
        on_speech_state: Callable[[bool], None] | None = None,
        threshold: float = 0.5,
        min_silence_ms: int = 400,
        preroll_ms: int = 300,
        min_speech_ms: int = 250,
        max_utterance_s: float = 18.0,
        hold_silence_ms: int = 700,
        lookback_ms: int = 2500,
        pause_window_ms: int = 96,
        speculate_after_ms: int = 64,
        speculate_interval_ms: int = 400,
        early_silence_ms: int = 240,
        on_provisional: Callable[[np.ndarray, int], None] | None = None,
    ):
        torch.set_num_threads(1)
        self._model = _shared_model()
        self._on_utterance = on_utterance
        self._on_speech_state = on_speech_state
        self._threshold = threshold

        self._min_silence_frames = max(1, min_silence_ms // FRAME_MS)
        self._min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self._max_frames = max(2, int(max_utterance_s * 1000) // FRAME_MS)
        self._hold_silence_frames = max(1, hold_silence_ms // FRAME_MS)
        self._lookback_frames = max(2, lookback_ms // FRAME_MS)
        self._pause_window = max(1, pause_window_ms // FRAME_MS)
        self._speculate_frames = max(1, speculate_after_ms // FRAME_MS)
        self._speculate_interval = max(1, speculate_interval_ms // FRAME_MS)
        self._early_silence_frames = max(1, early_silence_ms // FRAME_MS)
        self._on_provisional = on_provisional

        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, preroll_ms // FRAME_MS))
        self._speaking = False
        self._segment: list[np.ndarray] = []
        self._probs: list[float] = []
        self._silence_run = 0
        self._speech_frames = 0
        self._speculated = False
        self._frames_seen = 0
        self._last_speculation = -10**9
        # Set by the pipeline when a speculative transcript looks like a
        # finished sentence. Holds the speech-frame count it applies to, so a
        # later utterance can never inherit permission from an earlier one.
        self._early_ok_frames = -1
        # The mirror of the above: the frame count of an utterance whose guess
        # was still mid-clause, so its pause should run longer than usual.
        self._hold_frames = -1

    def reset(self) -> None:
        self._model.reset_states()
        self._preroll.clear()
        self._segment.clear()
        self._probs.clear()
        self._speaking = False
        self._silence_run = 0
        self._speech_frames = 0
        self._speculated = False
        self._early_ok_frames = -1
        self._hold_frames = -1

    # -- input ----------------------------------------------------------------

    def push(self, frame: np.ndarray) -> None:
        with torch.no_grad():
            prob = float(self._model(torch.from_numpy(frame), ASR_RATE).item())
        self._advance(frame, prob)

    def allow_early_commit(self, speech_frames: int) -> None:
        """Called from the recognise thread once a guess reads as a complete
        sentence. Lets the current pause end sooner than usual."""
        self._early_ok_frames = speech_frames

    def hold_commit(self, speech_frames: int) -> None:
        """Called from the recognise thread when a guess reads as an unfinished
        clause. Keeps the pause open longer, so a breath taken mid-sentence is
        not mistaken for the end of one.

        Best effort. The guess has to come back before the pause would
        otherwise have been confirmed, and on a long segment transcription
        sometimes takes longer than that. Missing the window costs nothing:
        the pause simply ends on the normal rule, which is what it did before
        this existed."""
        self._hold_frames = speech_frames

    def _advance(self, frame: np.ndarray, prob: float) -> None:
        """Split out from push() so tests can drive it without the model."""
        self._frames_seen += 1
        gate = self._threshold - HYSTERESIS if self._speaking else self._threshold
        is_speech = prob >= gate

        if not self._speaking:
            self._preroll.append(frame)
            if is_speech:
                self._speaking = True
                self._segment = list(self._preroll)
                # Preroll frames predate scoring; treat them as silence so they
                # are never chosen as a cut point.
                self._probs = [0.0] * (len(self._segment) - 1) + [prob]
                self._preroll.clear()
                self._silence_run = 0
                self._speech_frames = 1
                if self._on_speech_state:
                    self._on_speech_state(True)
            return

        self._segment.append(frame)
        self._probs.append(prob)
        if is_speech:
            self._silence_run = 0
            self._speech_frames += 1
            # New speech invalidates any guess made during the last gap.
            self._speculated = False
        else:
            self._silence_run += 1

        # A short way into a pause, hand the audio over to be transcribed
        # speculatively. Trailing silence does not change what Whisper hears
        # (measured identical at 0, 120, 250 and 400 ms), so a guess made now
        # is still correct when the pause is confirmed, and most of the
        # recognition cost disappears inside time the listener already waits.
        if (
            self._on_provisional is not None
            and not self._speculated
            and self._silence_run == self._speculate_frames
            and self._speech_frames >= self._min_speech_frames
            and self._frames_seen - self._last_speculation >= self._speculate_interval
        ):
            self._speculated = True
            self._last_speculation = self._frames_seen
            self._on_provisional(np.concatenate(self._segment), self._speech_frames)

        needed = self._min_silence_frames
        # A guess that already reads as a finished sentence means the pause
        # does not have to run its full length. Whisper's trailing punctuation
        # is a good signal for this, though not a perfect one, so it is only
        # trusted when the recogniser was also confident.
        if self._early_ok_frames == self._speech_frames:
            needed = min(needed, self._early_silence_frames)
        elif self._hold_frames == self._speech_frames:
            # Measured on the twelve enrollment clips, all of which are
            # complete sentences: Whisper supplied terminal punctuation for
            # eleven. So a hold fires wrongly on about one sentence in twelve
            # and costs 300 ms when it does, which averages out to roughly
            # 25 ms per sentence. Cheap enough to apply to every utterance
            # rather than only long ones.
            needed = max(needed, self._hold_silence_frames)

        if self._silence_run >= needed:
            self._flush(len(self._segment), final=True)
        elif len(self._segment) >= self._max_frames:
            self._flush(self._quietest_split(), final=False)

    # -- splitting ------------------------------------------------------------

    def _quietest_split(self) -> int:
        """Index to cut at when no pause has arrived.

        Looks back over the last few seconds for the run of frames with the
        lowest mean speech probability, and cuts at the middle of it. Falls back
        to the end of the buffer if the search window is degenerate.
        """
        total = len(self._segment)
        start = max(self._min_speech_frames, total - self._lookback_frames)
        if total - start <= self._pause_window:
            return total

        window = self._pause_window
        probs = np.asarray(self._probs, dtype=np.float32)
        # Mean probability of every candidate run of `window` frames.
        kernel = np.ones(window, dtype=np.float32) / window
        means = np.convolve(probs[start:], kernel, mode="valid")
        if means.size == 0:
            return total
        best = int(np.argmin(means)) + start
        return min(total, best + window // 2)

    def _flush(self, index: int, final: bool) -> None:
        """Emit segment[:index] and keep the rest as the start of the next one."""
        index = max(0, min(index, len(self._segment)))
        head, tail = self._segment[:index], self._segment[index:]
        head_probs, tail_probs = self._probs[:index], self._probs[index:]

        if self._speech_frames >= self._min_speech_frames and head:
            self._on_utterance(np.concatenate(head), self._speech_frames)

        # Permission to end early belongs to the utterance it was granted for.
        # Leaving it set let the next utterance inherit it as soon as its own
        # speech-frame count happened to match, which cut sentences short at
        # 240 ms with no guess ever made for them.
        self._early_ok_frames = -1
        self._hold_frames = -1
        self._speculated = False

        if final:
            self._segment = []
            self._probs = []
            self._speaking = False
            self._silence_run = 0
            self._speech_frames = 0
            self._model.reset_states()
            self._preroll.clear()
            if self._on_speech_state:
                self._on_speech_state(False)
            return

        # Mid-utterance split: everything after the cut carries forward so no
        # audio is lost, and speech accounting restarts against the remainder.
        self._segment = tail
        self._probs = tail_probs
        self._silence_run = 0
        self._speech_frames = sum(
            1 for p in tail_probs if p >= self._threshold - HYSTERESIS
        )
        del head_probs
