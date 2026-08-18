"""Segmentation rules, checked without a model or a microphone.

Drives the Segmenter's decision logic with synthetic voice-activity
probabilities and asserts where the cuts land. The recognise thread is modelled
too — a speculation fires, a transcript comes back some milliseconds later, and
it either reads as a finished sentence or as an unfinished clause — because
without that half the pause rules can never be exercised.

The case that matters most is the first one. An earlier version shortened the
required pause to 180 ms for anything past eight seconds, which meant a long
sentence ended at the speaker's next breath. Breaths between clauses measure
around 200 ms, so it split reliably and in the middle of a thought.

    .venv\\Scripts\\python.exe tools\\test_segmenter.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from talkbox.vad import FRAME_MS, Segmenter  # noqa: E402

FAILURES: list[str] = []


def run(plan, finished_at_end=True, asr_ms=280):
    """plan: (kind, ms) pairs, kind is 'speech' or 'gap'.

    Returns the times, in seconds, at which an utterance was emitted.
    """
    cuts: list[int] = []
    pending: list[list[int]] = []
    seg = Segmenter.__new__(Segmenter)
    Segmenter.__init__(
        seg,
        on_utterance=lambda audio, frames: cuts.append(len(audio)),
        on_provisional=lambda audio, frames: pending.append([asr_ms, frames]),
    )
    # The model is only consulted by push(); _advance takes the probability
    # directly, so a stub that can be reset is enough.
    seg._model = types.SimpleNamespace(reset_states=lambda: None)

    frame = np.zeros(512, dtype=np.float32)
    total = sum(ms for _, ms in plan)
    elapsed = 0
    marks: list[float] = []
    for kind, ms in plan:
        prob = 0.95 if kind == "speech" else 0.02
        for _ in range(max(1, ms // FRAME_MS)):
            for job in list(pending):
                job[0] -= FRAME_MS
                if job[0] <= 0:
                    pending.remove(job)
                    last = elapsed > total - 1200
                    if finished_at_end and last:
                        seg.allow_early_commit(job[1])
                    else:
                        seg.hold_commit(job[1])
            before = len(cuts)
            seg._advance(frame, prob)
            elapsed += FRAME_MS
            if len(cuts) > before:
                marks.append(elapsed / 1000.0)
    return marks


def check(name, ok, detail=""):
    print("  %-4s %s%s" % ("ok" if ok else "FAIL", name, "  " + detail if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    print("segmentation rules\n")

    plan = []
    for _ in range(9):
        plan += [("speech", 1800), ("gap", 200)]
    plan += [("gap", 600)]
    marks = run(plan)
    # 18 s of speech exceeds max_utterance_s, so one cut is expected. What must
    # not happen is a cut around 10 s, which is where the old soft rule fired.
    early = [m for m in marks if m < 15]
    check("a long sentence is not split at a 200 ms breath",
          not early, "cuts at %s" % ["%.2f" % m for m in marks])

    plan = []
    for _ in range(4):
        plan += [("speech", 2500), ("gap", 420)]
    plan += [("gap", 600)]
    marks = run(plan)
    check("a 420 ms thinking pause does not end the sentence",
          len(marks) == 1, "%d utterance(s)" % len(marks))

    marks = run([("speech", 2000), ("gap", 900)])
    check("a finished sentence still ends promptly",
          len(marks) == 1 and marks[0] < 2.5,
          "ended at %.2fs" % marks[0] if marks else "never ended")

    marks = run([("speech", 2000), ("gap", 1400)], finished_at_end=False)
    check("an unpunctuated sentence still ends, just later",
          len(marks) == 1 and 2.5 < marks[0] < 3.0,
          "ended at %.2fs" % marks[0] if marks else "never ended")

    marks = run([("speech", 25000)])
    check("unbroken speech is cut at the hard limit",
          len(marks) >= 1 and marks[0] < 19,
          "first cut at %.2fs" % marks[0] if marks else "never cut")

    # Permission granted to one utterance must not be inherited by the next.
    seg = Segmenter.__new__(Segmenter)
    Segmenter.__init__(seg, on_utterance=lambda a, f: None)
    seg._model = types.SimpleNamespace(reset_states=lambda: None)
    seg.allow_early_commit(42)
    seg._flush(0, final=True)
    check("early-commit permission does not survive a flush",
          seg._early_ok_frames == -1 and seg._hold_frames == -1)

    print()
    if FAILURES:
        print("%d failed: %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all segmentation rules hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
