"""Typing dictated text into whatever window has focus.

The speech side of dictation is the same pipeline the rest of the app uses:
voice activity detection, then recognition, then optionally translation. The
only thing that differs is where the finished sentence goes. Instead of being
handed to the voice model it is typed, character by character, into whichever
application the user is looking at.

Windows has no supported way to insert text into another process's edit
control directly, so this synthesises keystrokes with SendInput. Text is sent
as UTF-16 code units with KEYEVENTF_UNICODE rather than as virtual key codes,
which means the receiving application sees the characters the user actually
said regardless of their keyboard layout: a Japanese sentence types correctly
on a US keyboard, and an accented Spanish word does not become a dead key.

The one rule worth stating plainly: nothing is typed while this application
itself holds the foreground. Otherwise pressing Start would immediately dictate
into the app's own transcript box, and a user switching back to check on it
would find their words landing somewhere they did not intend.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_RETURN = 0x0D

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# ULONG_PTR is pointer-sized, and getting it wrong silently corrupts the
# structure on 64-bit, so it is derived rather than assumed.
ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


_user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
_user32.SendInput.restype = wintypes.UINT


def _key_event(scan: int, keyup: bool, unicode_char: bool = True) -> _INPUT:
    flags = KEYEVENTF_UNICODE if unicode_char else 0
    if keyup:
        flags |= KEYEVENTF_KEYUP
    event = _INPUT(type=INPUT_KEYBOARD)
    event.ki = _KEYBDINPUT(
        wVk=0 if unicode_char else scan,
        wScan=scan if unicode_char else 0,
        dwFlags=flags,
        time=0,
        dwExtraInfo=0,
    )
    return event


def owns_foreground() -> bool:
    """Is the window the user is typing into one of ours?"""
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return False
    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value == _kernel32.GetCurrentProcessId()


def foreground_title() -> str:
    """Name of the window text would currently be typed into."""
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return ""
    length = _user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


class Typist:
    """Types finished sentences into the focused window.

    Keeps track of whether a space is owed. Whisper returns each utterance
    without surrounding whitespace, so dictating two sentences in a row would
    otherwise run them together as "one sentence.Another sentence."
    """

    # Sent per batch. SendInput accepts an array, and passing a whole sentence
    # in one call is both faster and less likely to interleave with real
    # keystrokes than one call per character. Kept well under the point where
    # applications with slow input hooks start dropping events.
    BATCH = 24

    def __init__(self, delay_ms: int = 0):
        self._delay = delay_ms / 1000.0
        self._owes_space = False

    def reset(self) -> None:
        self._owes_space = False

    def type(self, text: str) -> bool:
        """Type one finished sentence. False if it was suppressed.

        Suppressed means our own window had focus, in which case the caller
        should still show the text but must not send it anywhere.
        """
        text = text.strip()
        if not text:
            return False
        if owns_foreground():
            return False

        if self._owes_space:
            text = " " + text
        self._owes_space = True
        self._send(text)
        return True

    def _send(self, text: str) -> None:
        """Synthesise the keystrokes for one string.

        Split out from type() so the encoding can be tested against a real
        edit control without the foreground guard refusing to run, which it
        would always do inside our own process.
        """
        # Encode to UTF-16 code units. Characters outside the basic plane
        # become surrogate pairs, and each half is sent as its own event,
        # which is what applications expect to receive.
        raw = text.encode("utf-16-le")
        units = [
            int.from_bytes(raw[i : i + 2], "little") for i in range(0, len(raw), 2)
        ]

        events: list[_INPUT] = []
        for unit in units:
            if unit == 0x000A or unit == 0x000D:
                events.append(_key_event(VK_RETURN, False, unicode_char=False))
                events.append(_key_event(VK_RETURN, True, unicode_char=False))
                continue
            events.append(_key_event(unit, False))
            events.append(_key_event(unit, True))

        for start in range(0, len(events), self.BATCH):
            batch = events[start : start + self.BATCH]
            array = (_INPUT * len(batch))(*batch)
            _user32.SendInput(len(batch), array, ctypes.sizeof(_INPUT))
            if self._delay:
                time.sleep(self._delay)
