"""Qt front end.

The pipeline runs on plain Python threads and never touches a widget. Anything
it wants to report goes through `Bridge`, whose signals Qt marshals onto the
GUI thread. That keeps Qt work out of the audio callbacks, where a stall is an
audible dropout.

Settings live in tabs rather than a collapsing panel. The panel version could
reveal more content than the window had room for, and Qt resolved that by
squashing controls to zero height instead of growing the window. Each tab also
scrolls, so a small screen scrolls rather than crushing anything.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import audio as audio_io
from . import languages
from . import quality
from .config import Config
from .enroll import (
    CLIP_PROMPTS,
    MIN_CLIPS,
    Recorder,
    inspect,
    inspect_set,
    save_profile_clips,
)
from .pipeline import Events, Pipeline
from .tts import BUILTIN_LABELS, BUILTIN_ORDER

CABLE_URL = "https://vb-audio.com/Cable/"

STYLE = """
QWidget { background: #16191f; color: #eceef2; font-size: 13px; }
QGroupBox {
    background: #1b1f26; border: 1px solid #333945; border-radius: 8px;
    margin-top: 14px; padding: 12px 10px 10px 10px; font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #aab2c0; }
QComboBox, QLineEdit, QDoubleSpinBox {
    background: #1d2027; border: 1px solid #2f343d; border-radius: 6px;
    padding: 5px 8px; min-height: 20px;
}
QComboBox:disabled, QLineEdit:disabled { color: #5c6371; }
QComboBox QAbstractItemView { background: #1d2027; selection-background-color: #2d6cdf; }
QPushButton {
    background: #262b34; border: 1px solid #363c47; border-radius: 6px;
    padding: 7px 14px; font-weight: 600;
}
QPushButton:hover { background: #2f3641; }
QPushButton:disabled { color: #5c6371; background: #1b1e24; }
QPushButton#primary { background: #2d6cdf; border-color: #2d6cdf; color: white; }
QPushButton#primary:hover { background: #3b7cf0; }
QPushButton#danger { background: #7a2734; border-color: #93303f; color: #ffd9de; }
QPushButton#link {
    background: transparent; border: none; color: #6ea8fe;
    padding: 2px 6px; font-weight: 600; text-decoration: underline;
}
QPushButton#link:hover { color: #9cc4ff; background: transparent; }
QTabWidget::pane { border: 1px solid #2a2e36; border-radius: 8px; top: -1px; }
QTabBar::tab {
    background: #1a1d23; color: #8b93a3; border: 1px solid #2a2e36;
    border-bottom: none; border-top-left-radius: 7px; border-top-right-radius: 7px;
    padding: 8px 20px; margin-right: 2px; font-weight: 600;
}
QTabBar::tab:selected { background: #232830; color: #e6e8ec; }
QTabBar::tab:hover { color: #e6e8ec; }
QTextEdit { background: #101216; border: 1px solid #2a2e36; border-radius: 8px; }
QProgressBar { background: #1d2027; border: none; border-radius: 4px; height: 8px; }
QProgressBar::chunk { background: #3fb950; border-radius: 4px; }
QSlider::groove:horizontal { height: 4px; background: #2f343d; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #7d8697; width: 14px; margin: -6px 0; border-radius: 7px;
}
QScrollArea { border: none; background: transparent; }
QLabel { background: transparent; }
QLabel#hint { color: #99a2b2; font-size: 11px; }
QLabel#stat { color: #8b93a3; font-family: Consolas, monospace; }
QFrame#sep { background: #333945; max-height: 1px; border: none; }
QRadioButton, QCheckBox { spacing: 8px; padding: 3px 0; background: transparent; }
QRadioButton:disabled, QCheckBox:disabled { color: #626a79; }
QRadioButton::indicator, QCheckBox::indicator {
    width: 15px; height: 15px; border: 2px solid #5b6472; background: #12151a;
}
QRadioButton::indicator { border-radius: 9px; }
QCheckBox::indicator { border-radius: 4px; }
QRadioButton::indicator:hover, QCheckBox::indicator:hover { border-color: #8892a3; }
QRadioButton::indicator:checked {
    border-color: #4d8dff;
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
        stop:0 #4d8dff, stop:0.5 #4d8dff, stop:0.55 #12151a, stop:1 #12151a);
}
QCheckBox::indicator:checked { border-color: #4d8dff; background: #4d8dff; }
QRadioButton::indicator:disabled, QCheckBox::indicator:disabled {
    border-color: #3a4049; background: #171a20;
}
QCheckBox::indicator:checked:disabled { border-color: #3f4a5e; background: #34405a; }
QRadioButton::indicator:checked:disabled {
    border-color: #3f4a5e;
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
        stop:0 #55617a, stop:0.5 #55617a, stop:0.55 #171a20, stop:1 #171a20);
}
"""

OK, WARN, BAD, PENDING = "ok", "warn", "bad", "pending"
_MARK = {
    OK: ("✓", "#3fb950"),
    WARN: ("⚠", "#d29922"),
    BAD: ("✗", "#f0736a"),
    PENDING: ("·", "#6e7681"),
}


def tts_default() -> str:
    from .tts import BUILTIN_DEFAULT

    return BUILTIN_DEFAULT


def hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("hint")
    label.setWordWrap(True)
    return label


def scrollable(inner: QWidget) -> QScrollArea:
    """Wrap a tab page so a short window scrolls instead of squashing it."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setWidget(inner)
    return area


class Bridge(QObject):
    status = Signal(str)
    ready = Signal()
    source = Signal(str)
    translation = Signal(str)
    latency = Signal(float)
    level = Signal(float)
    speech = Signal(bool)
    error = Signal(str)
    busy = Signal(int, int)
    unclear = Signal(str)
    talkover = Signal()
    dictation = Signal(str, bool)
    selftest = Signal(bool, str, str)


class StatusRow(QWidget):
    """One line of the setup checklist: mark, label, value, optional action."""

    def __init__(self, title: str):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(10)

        self._mark = QLabel("·")
        self._mark.setFixedWidth(14)
        self._title = QLabel(title)
        self._title.setFixedWidth(88)
        self._title.setStyleSheet("color: #8b93a3;")
        self._value = QLabel("...")
        self._value.setWordWrap(True)
        self._action = QPushButton("")
        self._action.setObjectName("link")
        self._action.hide()
        self._handler = None

        layout.addWidget(self._mark)
        layout.addWidget(self._title)
        layout.addWidget(self._value, 1)
        layout.addWidget(self._action)

    def set(self, state: str, value: str, action: str = "", on_action=None) -> None:
        glyph, colour = _MARK[state]
        self._mark.setText(glyph)
        self._mark.setStyleSheet(f"color: {colour}; font-weight: 700;")
        self._value.setText(value)
        self._value.setStyleSheet(
            "color: #e6e8ec;" if state == OK else f"color: {colour};"
        )
        if self._handler is not None:
            self._action.clicked.disconnect(self._handler)
            self._handler = None
        if action and on_action is not None:
            self._action.setText(action)
            self._handler = on_action
            self._action.clicked.connect(on_action)
            self._action.show()
        else:
            self._action.hide()


class Overlay(QWidget):
    """Frameless, always on top, drag anywhere. Shows the translation."""

    def __init__(self):
        super().__init__(
            None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(880, 150)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 20, 26, 20)
        layout.setSpacing(8)

        # Only appears for speech that was too unclear to say out loud.
        self._src = QLabel("")
        self._src.setWordWrap(True)
        self._src.setStyleSheet(
            "color: #8b93a3; font-size: 15px; background: transparent;"
        )
        self._src.hide()

        self._dst = QLabel("Waiting for speech...")
        self._dst.setWordWrap(True)
        font = QFont()
        font.setPointSize(20)
        font.setWeight(QFont.DemiBold)
        self._dst.setFont(font)
        self._dst.setStyleSheet("color: #ffffff; background: transparent;")

        layout.addWidget(self._src)
        layout.addWidget(self._dst)
        self._drag = None

    def set_source(self, text: str) -> None:
        self._src.setText(text)
        self._src.setVisible(bool(text) and text.startswith("(unclear)"))

    def set_translation(self, text: str) -> None:
        self._dst.setText(text)
        self._src.hide()

    def paintEvent(self, _event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(0, 0, -1, -1), 14, 14)
        painter.fillPath(path, QColor(10, 12, 16, 226))
        painter.strokePath(path, QColor(70, 78, 92, 170))

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, _event):  # noqa: N802
        self._drag = None


class EnrollDialog(QDialog):
    """Records the reference clips, one prompt at a time."""

    def __init__(self, parent, device: audio_io.DeviceInfo, existing: list[str]):
        super().__init__(parent)
        self.setWindowTitle("Record your voice")
        self.setMinimumWidth(660)
        self.setStyleSheet(STYLE)

        self._device = device
        self._existing = existing
        self._recorder: Recorder | None = None
        self._clips: list[tuple | None] = [None] * len(CLIP_PROMPTS)
        self._index = 0
        self.profile_name: str | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(
            hint(
                f"{len(CLIP_PROMPTS)} short clips, about five seconds each. Read "
                f"each one the way it is labelled. At least {MIN_CLIPS} to save, "
                f"all {len(CLIP_PROMPTS)} for the best result.\n"
                f"Recording from: {device.short}"
            )
        )

        head = QHBoxLayout()
        self._step_label = QLabel("")
        font = QFont()
        font.setPointSize(11)
        font.setWeight(QFont.DemiBold)
        self._step_label.setFont(font)
        self._dots = QLabel("")
        self._dots.setObjectName("stat")
        head.addWidget(self._step_label, 1)
        head.addWidget(self._dots)
        layout.addLayout(head)

        self._passage = QTextEdit()
        self._passage.setReadOnly(True)
        self._passage.setFixedHeight(96)
        self._passage.setStyleSheet(
            "background:#101216; border:1px solid #2a2e36; border-radius:8px;"
            "font-size:16px; line-height:150%; padding:10px;"
        )
        layout.addWidget(self._passage)

        meters = QGridLayout()
        meters.addWidget(QLabel("Level"), 0, 0)
        self._level = QProgressBar()
        self._level.setRange(0, 100)
        self._level.setTextVisible(False)
        meters.addWidget(self._level, 0, 1)
        self._timer_label = QLabel("0.0 s")
        self._timer_label.setObjectName("stat")
        meters.addWidget(self._timer_label, 0, 2)
        layout.addLayout(meters)

        self._feedback = hint("")
        layout.addWidget(self._feedback)

        nav = QHBoxLayout()
        self._record_btn = QPushButton("Start recording")
        self._record_btn.setObjectName("primary")
        self._record_btn.setMinimumWidth(150)
        self._record_btn.clicked.connect(self._toggle)
        self._back_btn = QPushButton("Back")
        self._back_btn.clicked.connect(lambda: self._go(self._index - 1))
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(lambda: self._go(self._index + 1))
        nav.addWidget(self._record_btn)
        nav.addStretch(1)
        nav.addWidget(self._back_btn)
        nav.addWidget(self._next_btn)
        layout.addLayout(nav)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Profile name"))
        self._name = QLineEdit(existing[0] if existing else "default")
        name_row.addWidget(self._name, 1)
        layout.addLayout(name_row)

        self._summary = hint("")
        layout.addWidget(self._summary)

        buttons = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("primary")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(self._save_btn)
        layout.addLayout(buttons)

        self._tick = QTimer(self)
        self._tick.setInterval(100)
        self._tick.timeout.connect(self._update_timer)
        self._peak = 0.0
        self._rms = 0.0
        self._go(0)

    def _dot_text(self) -> str:
        return " ".join("●" if c else "○" for c in self._clips)

    def _go(self, index: int) -> None:
        if self._recorder is not None:
            self._stop_recording()
        self._index = max(0, min(len(CLIP_PROMPTS) - 1, index))
        title, passage = CLIP_PROMPTS[self._index]
        self._step_label.setText(
            "Clip %d of %d: %s" % (self._index + 1, len(CLIP_PROMPTS), title)
        )
        self._passage.setPlainText(passage)
        self._dots.setText(self._dot_text())
        self._back_btn.setEnabled(self._index > 0)
        self._next_btn.setEnabled(self._index < len(CLIP_PROMPTS) - 1)
        done = self._clips[self._index]
        self._record_btn.setText("Re-record" if done else "Start recording")
        self._timer_label.setText(
            "%0.1f s" % (done[0].size / done[1]) if done else "0.0 s"
        )
        self._level.setValue(0)
        self._feedback.setText("")
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        recorded = [c for c in self._clips if c]
        if recorded:
            ok, message = inspect_set(recorded)
        else:
            ok, message = False, "Nothing recorded yet."
        total = sum(a.size / r for a, r in recorded)
        self._summary.setText(
            "%d of %d clips, %.0fs. %s"
            % (len(recorded), len(CLIP_PROMPTS), total, message)
        )
        self._save_btn.setEnabled(ok)

    def _on_level(self, rms: float, peak: float) -> None:
        """PortAudio thread. Stores only; the timer paints it."""
        self._peak = max(self._peak * 0.97, peak)
        self._rms = rms

    def _toggle(self) -> None:
        if self._recorder is None:
            self._peak = 0.0
            self._recorder = Recorder(self._device, on_level=self._on_level)
            try:
                self._recorder.start()
            except Exception as exc:
                self._recorder = None
                QMessageBox.critical(self, "Microphone", f"Could not open input: {exc}")
                return
            self._record_btn.setText("Stop")
            self._feedback.setText("Recording. Read the line above.")
            self._tick.start()
        else:
            self._stop_recording()

    def _stop_recording(self) -> None:
        if self._recorder is None:
            return
        self._tick.stop()
        rate = self._recorder.rate
        audio = self._recorder.stop()
        self._recorder = None
        self._level.setValue(0)
        self._record_btn.setText("Re-record")

        ok, message = inspect(audio, rate, minimum=3.0)
        self._feedback.setText(message)
        self._clips[self._index] = (audio, rate) if ok else None
        self._dots.setText(self._dot_text())
        self._refresh_summary()

    def _update_timer(self) -> None:
        if self._recorder is None:
            return
        secs = self._recorder.seconds
        self._timer_label.setText("%0.1f s" % secs)
        self._level.setValue(int(min(1.0, self._rms * 6.0) * 100))
        if self._peak >= 0.999:
            self._feedback.setText("Too loud. Lower the input gain and redo this clip.")
        elif secs >= 8:
            self._feedback.setText("That is enough for this clip.")

    def _save(self) -> None:
        name = self._name.text().strip() or "default"
        name = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
        if not name:
            QMessageBox.warning(self, "Name", "Pick a valid profile name.")
            return
        if name in self._existing:
            reply = QMessageBox.question(
                self,
                "Replace recording",
                "Profile '%s' already exists. Replace it?" % name,
            )
            if reply != QMessageBox.Yes:
                return
        try:
            save_profile_clips(name, [c for c in self._clips if c])
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.profile_name = name
        self.accept()

    def reject(self) -> None:
        self._tick.stop()
        if self._recorder is not None:
            self._recorder.stop()
            self._recorder = None
        super().reject()


# --------------------------------------------------------------------------- #
#  Main window
# --------------------------------------------------------------------------- #


class MainWindow(QWidget):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Talkbox")
        self.setStyleSheet(STYLE)
        self.setMinimumSize(720, 560)

        self.bridge = Bridge()
        self.pipeline = Pipeline(
            cfg,
            Events(
                on_status=self.bridge.status.emit,
                on_ready=self.bridge.ready.emit,
                on_source=self.bridge.source.emit,
                on_translation=self.bridge.translation.emit,
                on_latency=self.bridge.latency.emit,
                on_level=self.bridge.level.emit,
                on_speech=self.bridge.speech.emit,
                on_error=self.bridge.error.emit,
                on_busy=self.bridge.busy.emit,
                on_unclear=self.bridge.unclear.emit,
                on_talkover=self.bridge.talkover.emit,
                on_dictation=self.bridge.dictation.emit,
            ),
        )
        self.overlay = Overlay()

        self._inputs: list[audio_io.DeviceInfo] = []
        self._outputs: list[audio_io.DeviceInfo] = []
        self._mic: audio_io.DeviceInfo | None = None
        self._monitor: audio_io.DeviceInfo | None = None
        self._cable: audio_io.DeviceInfo | None = None
        self._has_voices = False
        self._latency_ms = 0.0
        self._queues = (0, 0)
        self._models_ready = False

        self._build()
        self._connect()
        self._rescan()
        self._load_settings()

        if not self._ensure_licence():
            self._on_status("Licence declined. Speech synthesis is unavailable.")
            self.record_btn.setEnabled(False)
            return

        # First run: fetch the models and walk through the setup. Doing this
        # before loading anything avoids a long silent wait on a fresh clone.
        from .wizard import needs_setup

        if needs_setup(cfg):
            self._run_wizard(first_run=True)

        self.bridge.status.emit("Loading models. First run takes a few minutes.")
        threading.Thread(target=self.pipeline.load_models, daemon=True).start()

    # -- layout ---------------------------------------------------------------

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        self.tabs = QTabWidget()
        self.tabs.addTab(scrollable(self._tab_talk()), "Talk")
        self.tabs.addTab(scrollable(self._tab_voice()), "Voice")
        self.tabs.addTab(scrollable(self._tab_audio()), "Audio")
        root.addWidget(self.tabs, 1)

        # Always visible, whichever tab is open.
        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primary")
        self.start_btn.setEnabled(False)
        self.start_btn.setFixedHeight(40)
        self.start_btn.setMinimumWidth(130)
        self.vu = QProgressBar()
        self.vu.setRange(0, 100)
        self.vu.setTextVisible(False)
        self.vu.setFixedWidth(120)
        self.stat_label = QLabel("")
        self.stat_label.setObjectName("stat")

        controls.addWidget(self.start_btn)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Mic"))
        controls.addWidget(self.vu)
        controls.addStretch(1)
        controls.addWidget(self.stat_label)
        root.addLayout(controls)

        self.status_label = hint("Starting up...")
        root.addWidget(self.status_label)

    def _tab_talk(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        mode_box = QGroupBox("What happens when you talk")
        mode = QVBoxLayout(mode_box)
        mode.setSpacing(6)
        self.speak_radio = QRadioButton("Speak it out loud, into the call")
        self.dictate_radio = QRadioButton("Type it into whatever window I'm in")
        mode.addWidget(self.speak_radio)
        mode.addWidget(self.dictate_radio)
        dict_row = QHBoxLayout()
        dict_row.addSpacing(22)
        self.dictate_translate_check = QCheckBox("Translate it before typing")
        dict_row.addWidget(self.dictate_translate_check)
        dict_row.addStretch(1)
        mode.addLayout(dict_row)
        self.dictate_hint = hint("")
        mode.addWidget(self.dictate_hint)
        layout.addWidget(mode_box)

        lang_box = QGroupBox("Languages")
        lang = QVBoxLayout(lang_box)
        lang.setSpacing(8)
        pair = QHBoxLayout()
        self.src_combo = QComboBox()
        self.dst_combo = QComboBox()
        for combo in (self.src_combo, self.dst_combo):
            combo.addItems(languages.NAMES)
            # A combo's minimumSizeHint covers its longest item, which stops a
            # row of them shrinking with the window. setMinimumContentsLength
            # does not lower that hint; an Ignored horizontal policy does, and
            # the explicit minimum then sets a sensible floor. Matters most for
            # the device combos, where names like "CABLE Output (VB-Audio
            # Virtual Cable)" are genuinely long.
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            combo.setMinimumWidth(96)
        self.swap_btn = QPushButton("⇄")
        self.swap_btn.setFixedWidth(44)
        self.auto_check = QCheckBox("Detect who is speaking")
        pair.addWidget(QLabel("I speak"))
        pair.addWidget(self.src_combo, 1)
        pair.addWidget(self.swap_btn)
        pair.addWidget(QLabel("They hear"))
        pair.addWidget(self.dst_combo, 1)
        lang.addLayout(pair)
        lang.addWidget(self.auto_check)
        layout.addWidget(lang_box)

        qual_box = QGroupBox("Quality")
        qual = QVBoxLayout(qual_box)
        qual.setSpacing(6)
        row = QHBoxLayout()
        self.quality_combo = QComboBox()
        for preset in quality.SPEECH_PRESETS:
            self.quality_combo.addItem(preset.label, preset.key)
        row.addWidget(QLabel("Level"))
        row.addWidget(self.quality_combo, 1)
        qual.addLayout(row)
        self.quality_hint = hint("")
        qual.addWidget(self.quality_hint)
        layout.addWidget(qual_box)

        setup_box = QGroupBox("Setup")
        setup = QVBoxLayout(setup_box)
        setup.setSpacing(0)
        self.row_mic = StatusRow("Microphone")
        self.row_hear = StatusRow("You hear")
        self.row_call = StatusRow("Call hears")
        self.row_voice = StatusRow("Your voice")
        for row_widget in (self.row_mic, self.row_hear, self.row_call, self.row_voice):
            setup.addWidget(row_widget)
        sep = QFrame()
        sep.setObjectName("sep")
        sep.setFrameShape(QFrame.HLine)
        setup.addSpacing(6)
        setup.addWidget(sep)
        setup.addSpacing(6)
        self.routing_note = hint("")
        setup.addWidget(self.routing_note)
        layout.addWidget(setup_box)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log.setMinimumHeight(120)
        layout.addWidget(self.log, 1)
        return page

    def _tab_voice(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        box = QGroupBox("Which voice speaks the translation")
        form = QVBoxLayout(box)
        form.setSpacing(8)

        self.clone_radio = QRadioButton("My own voice")
        self.builtin_radio = QRadioButton("A built-in voice")
        form.addWidget(self.clone_radio)

        row = QHBoxLayout()
        row.addSpacing(22)
        self.voice_label = QLabel("")
        self.record_btn = QPushButton("Record my voice...")
        row.addWidget(self.voice_label, 1)
        row.addWidget(self.record_btn)
        form.addLayout(row)
        why = QHBoxLayout()
        why.addSpacing(22)
        why.addWidget(
            hint(
                f"{len(CLIP_PROMPTS)} short clips covering different delivery. "
                "Record somewhere quiet: any background noise ends up in every "
                "sentence the app speaks."
            ),
            1,
        )
        form.addLayout(why)

        form.addSpacing(6)
        form.addWidget(self.builtin_radio)
        picker = QHBoxLayout()
        picker.addSpacing(22)
        self.builtin_combo = QComboBox()
        for full in BUILTIN_ORDER:
            self.builtin_combo.addItem(BUILTIN_LABELS[full], full)
        self.builtin_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.builtin_combo.setMinimumContentsLength(12)
        self.builtin_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.builtin_combo.setMinimumWidth(140)
        picker.addWidget(self.builtin_combo, 1)
        form.addLayout(picker)
        note = QHBoxLayout()
        note.addSpacing(22)
        note.addWidget(
            hint(
                "No recording needed. These ship inside the voice model and "
                "speak every language it supports, same as a clone does."
            ),
            1,
        )
        form.addLayout(note)
        layout.addWidget(box)

        layout.addStretch(1)
        return page

    def _tab_audio(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        box = QGroupBox("Devices")
        form = QFormLayout(box)
        form.setSpacing(8)
        self.auto_devices_check = QCheckBox("Pick devices automatically")
        form.addRow(self.auto_devices_check)
        self.input_combo = QComboBox()
        form.addRow("Microphone", self.input_combo)
        self.output_combo = QComboBox()
        form.addRow("Send to call via", self.output_combo)
        for combo in (self.input_combo, self.output_combo):
            combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            combo.setMinimumWidth(120)
        form.addRow("", hint("This must be a virtual cable for a call to hear it."))
        self.monitor_check = QCheckBox("Play the translation to me as well")
        self.monitor_combo = QComboBox()
        form.addRow(self.monitor_check, self.monitor_combo)
        form.addRow("", hint("Off by default. Hearing yourself talk is distracting."))
        layout.addWidget(box)

        guard_box = QGroupBox("Feedback")
        guard = QVBoxLayout(guard_box)
        self.duplex_check = QCheckBox("Ignore the mic while the translation speaks")
        guard.addWidget(self.duplex_check)
        guard.addWidget(
            hint(
                "Stops the mic picking up its own output. Set automatically "
                "from your output device. Ticking it here pins your choice."
            )
        )
        layout.addWidget(guard_box)

        extras_box = QGroupBox("Extras")
        extras = QVBoxLayout(extras_box)
        self.overlay_check = QCheckBox("Show the subtitle bar on screen")
        self.transcript_check = QCheckBox("Save a transcript to disk")
        extras.addWidget(self.overlay_check)
        extras.addWidget(self.transcript_check)
        layout.addWidget(extras_box)

        actions = QHBoxLayout()
        self.rescan_btn = QPushButton("Rescan devices")
        self.selftest_btn = QPushButton("Test my setup")
        self.selftest_btn.setEnabled(False)
        self.wizard_btn = QPushButton("Run setup again")
        actions.addWidget(self.rescan_btn)
        actions.addWidget(self.selftest_btn)
        actions.addWidget(self.wizard_btn)
        actions.addStretch(1)
        layout.addLayout(actions)
        layout.addWidget(
            hint("Test plays a sentence into the cable and records it back.")
        )
        layout.addStretch(1)
        return page

    # -- wiring ---------------------------------------------------------------

    def _connect(self) -> None:
        self.bridge.status.connect(self._on_status)
        self.bridge.ready.connect(self._on_ready)
        self.bridge.error.connect(self._on_error)
        self.bridge.source.connect(self._on_source)
        self.bridge.translation.connect(self._on_translation)
        self.bridge.latency.connect(self._on_latency)
        self.bridge.level.connect(self._on_level)
        self.bridge.busy.connect(self._on_busy)
        self.bridge.unclear.connect(self._on_unclear)
        self.bridge.talkover.connect(self._on_talkover)
        self.bridge.dictation.connect(self._on_dictation)
        self.bridge.selftest.connect(self._on_selftest_done)

        self.start_btn.clicked.connect(self._toggle)
        self.record_btn.clicked.connect(self._record_voice)
        self.clone_radio.toggled.connect(self._on_voice_mode)
        self.speak_radio.toggled.connect(self._on_mode)
        self.dictate_translate_check.toggled.connect(self._on_mode_hint)
        self.rescan_btn.clicked.connect(self._rescan_clicked)
        self.selftest_btn.clicked.connect(self._run_selftest)
        self.wizard_btn.clicked.connect(self._run_wizard)
        self.swap_btn.clicked.connect(self._swap)
        self.auto_devices_check.toggled.connect(self._on_auto_devices)
        self.monitor_check.toggled.connect(self._on_monitor_toggled)
        self.overlay_check.toggled.connect(self._toggle_overlay)
        self.quality_combo.currentIndexChanged.connect(self._on_quality)
        self.duplex_check.toggled.connect(self._on_duplex)
        for combo in (self.src_combo, self.dst_combo):
            combo.currentIndexChanged.connect(self._push_direction)
        self.auto_check.toggled.connect(self._push_direction)
        for combo in (self.input_combo, self.output_combo, self.monitor_combo):
            combo.currentIndexChanged.connect(self._on_manual_device)

    # -- devices ---------------------------------------------------------------

    def _rescan_clicked(self) -> None:
        self._rescan()
        self._on_status("Devices rescanned.")

    def _rescan(self) -> None:
        self._inputs = audio_io.input_devices()
        self._outputs = audio_io.output_devices()

        for combo, devices in (
            (self.input_combo, self._inputs),
            (self.output_combo, self._outputs),
            (self.monitor_combo, self._outputs),
        ):
            combo.blockSignals(True)
            combo.clear()
            for dev in devices:
                combo.addItem(dev.label, dev)
            combo.blockSignals(False)

        manual = not self.cfg.auto_devices and self.cfg.input_device_name
        if manual:
            self._mic = audio_io.resolve_device(
                self._inputs, self.cfg.input_device, self.cfg.input_device_name
            )
            self._monitor = audio_io.resolve_device(
                self._outputs, self.cfg.monitor_device, self.cfg.monitor_device_name
            )
            self._cable = audio_io.resolve_device(
                self._outputs, self.cfg.output_device, self.cfg.output_device_name
            )
        else:
            self._mic = audio_io.autoselect_input(self._inputs)
            self._monitor = audio_io.autoselect_monitor(self._outputs)
            self._cable = audio_io.autoselect_call_sink(self._outputs)

        self._sync_device_combos()
        self._refresh_voices()
        self._refresh_setup()

    def _sync_device_combos(self) -> None:
        for combo, dev in (
            (self.input_combo, self._mic),
            (self.output_combo, self._cable or self._monitor),
            (self.monitor_combo, self._monitor),
        ):
            if dev is None:
                continue
            combo.blockSignals(True)
            for i in range(combo.count()):
                if combo.itemData(i).index == dev.index:
                    combo.setCurrentIndex(i)
                    break
            combo.blockSignals(False)

    def _refresh_setup(self) -> None:
        """Repaint the checklist.

        Each row is one short value. Anything longer, like which device the
        call application has to be pointed at, goes in the note underneath,
        because a long value wraps under its own label and looks broken.
        """
        if self._mic is None:
            self.row_mic.set(BAD, "None found")
        else:
            self.row_mic.set(OK, self._mic.short)

        if not self.cfg.monitor_enabled:
            self.row_hear.set(PENDING, "Off")
        elif self._monitor is None:
            self.row_hear.set(WARN, "None found")
        else:
            self.row_hear.set(OK, self._monitor.short)

        if self._cable is not None and not audio_io.is_cable_sink(self._cable):
            self.row_call.set(
                WARN,
                "Nothing. %s is a normal output." % self._cable.short,
                action="Fix",
                on_action=self._show_cable_help,
            )
            self.routing_note.setText(
                "A call cannot listen to a normal output. Install a virtual "
                "cable to send the translation to one."
            )
        elif self._cable is not None:
            source = audio_io.call_source_name(self._cable)
            short_source = source.split(" (")[0]
            self.row_call.set(OK, self._cable.short)
            self.routing_note.setText(
                "In your call app, set the microphone to %s.\n"
                "Only the translation is sent. Your own voice never reaches "
                "the call." % short_source
            )
        else:
            self.row_call.set(
                BAD,
                "Nothing yet",
                action="Fix",
                on_action=self._show_cable_help,
            )
            self.routing_note.setText(
                "No virtual cable installed, so a call cannot hear the "
                "translation. Everything else works."
            )

        recorded = (
            "Profile: %s" % self.cfg.speaker_profile
            if self._has_voices
            else "No recording yet"
        )
        if self.builtin_radio.isChecked():
            self.row_voice.set(OK, self.builtin_combo.currentText())
            self.voice_label.setText(recorded)
        elif self._has_voices:
            # Not "Your voice": that is the row's own label, and repeating it
            # as the value read as a placeholder someone forgot to fill in.
            self.row_voice.set(OK, "Recorded")
            self.voice_label.setText(recorded)
        else:
            self.row_voice.set(
                BAD, "Not recorded", action="Record", on_action=self._record_voice
            )
            self.voice_label.setText(recorded)

        self._update_start_enabled()

    def _update_start_enabled(self) -> None:
        ready = self._models_ready and self._voice_ready() and self._mic is not None
        self.start_btn.setEnabled(ready or self.pipeline.running)
        self.selftest_btn.setEnabled(
            self._models_ready and self._voice_ready() and not self.pipeline.running
        )

    def _show_cable_help(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Let calls hear the translation")
        box.setTextFormat(Qt.RichText)
        box.setText("<b>Install a virtual microphone</b>")
        box.setInformativeText(
            "Windows cannot feed audio to Discord, Zoom or Teams as a "
            "microphone without a virtual audio driver.<br><br>"
            "1. Download VB-Cable and unzip it.<br>"
            "2. Right-click <b>VBCABLE_Setup_x64.exe</b>, Run as "
            "administrator.<br>"
            "3. Reboot.<br>"
            "4. Press <b>Rescan devices</b> here.<br>"
            "5. Set your call app microphone to <b>CABLE Output</b>."
        )
        open_btn = box.addButton("Open download page", QMessageBox.AcceptRole)
        box.addButton("Close", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl(CABLE_URL))

    def _on_auto_devices(self, on: bool) -> None:
        self.cfg.auto_devices = on
        for widget in (self.input_combo, self.output_combo):
            widget.setEnabled(not on and not self.pipeline.running)
        self.monitor_combo.setEnabled(
            not on and not self.pipeline.running and self.monitor_check.isChecked()
        )
        if on:
            self._rescan()

    def _on_manual_device(self) -> None:
        if self.cfg.auto_devices or self.pipeline.running:
            return
        self._mic = self.input_combo.currentData()
        self._cable = self.output_combo.currentData()
        self._monitor = self.monitor_combo.currentData()
        self._refresh_setup()

    def _on_monitor_toggled(self, on: bool) -> None:
        self.cfg.monitor_enabled = on
        self.monitor_combo.setEnabled(
            on and not self.cfg.auto_devices and not self.pipeline.running
        )
        self._refresh_setup()

    # -- settings --------------------------------------------------------------

    def _load_settings(self) -> None:
        cfg = self.cfg
        idx = self.quality_combo.findData(cfg.speech_quality)
        self.quality_combo.setCurrentIndex(idx if idx >= 0 else 1)

        for widget, value in (
            (self.auto_devices_check, cfg.auto_devices),
            (self.monitor_check, cfg.monitor_enabled),
            (self.duplex_check, cfg.duplex_guard),
            (self.overlay_check, cfg.show_overlay),
            (self.transcript_check, cfg.save_transcript),
        ):
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)

        dictating = cfg.mode == "dictate"
        for widget in (self.speak_radio, self.dictate_radio):
            widget.blockSignals(True)
        self.speak_radio.setChecked(not dictating)
        self.dictate_radio.setChecked(dictating)
        for widget in (self.speak_radio, self.dictate_radio):
            widget.blockSignals(False)
        self.dictate_translate_check.blockSignals(True)
        self.dictate_translate_check.setChecked(cfg.dictate_translate)
        self.dictate_translate_check.blockSignals(False)
        self._on_mode(not dictating)

        builtin = cfg.voice_mode == "builtin"
        for widget in (self.clone_radio, self.builtin_radio):
            widget.blockSignals(True)
        self.clone_radio.setChecked(not builtin)
        self.builtin_radio.setChecked(builtin)
        for widget in (self.clone_radio, self.builtin_radio):
            widget.blockSignals(False)
        pick = self.builtin_combo.findData(cfg.builtin_voice)
        if pick < 0:
            pick = self.builtin_combo.findData(tts_default())
        self.builtin_combo.setCurrentIndex(max(0, pick))
        self._on_voice_mode(not builtin)

        self.src_combo.setCurrentText(languages.get(cfg.source_lang).name)
        self.dst_combo.setCurrentText(languages.get(cfg.target_lang).name)
        self._refresh_quality_hints()
        self._on_auto_devices(cfg.auto_devices)
        self._on_monitor_toggled(cfg.monitor_enabled)

    def _on_mode(self, speaking: bool) -> None:
        """Show only what the chosen mode uses."""
        running = self.pipeline.running
        self.dictate_translate_check.setEnabled(not speaking and not running)
        self.tabs.setTabEnabled(1, speaking)  # Voice
        self._on_mode_hint()
        self._refresh_setup()

    def _on_mode_hint(self) -> None:
        if self.speak_radio.isChecked():
            self.dictate_hint.setText(
                "The translation is spoken in your voice and sent to the call."
            )
            return
        if self.dictate_translate_check.isChecked():
            self.dictate_hint.setText(
                "Speak one language, the other one gets typed. Click into the "
                "window you want the text in first: nothing is typed while "
                "this window is in front."
            )
        else:
            self.dictate_hint.setText(
                "What you say gets typed out as you say it. Click into the "
                "window you want the text in first: nothing is typed while "
                "this window is in front."
            )

    def _on_voice_mode(self, cloning: bool) -> None:
        """Grey out whichever half of the tab does not apply."""
        running = self.pipeline.running
        self.record_btn.setEnabled(cloning and not running)
        self.voice_label.setEnabled(cloning)
        self.builtin_combo.setEnabled(not cloning and not running)
        self._refresh_setup()

    def _voice_ready(self) -> bool:
        """A built-in voice is always ready; a cloned one needs a recording.

        Dictation needs neither, since nothing is ever spoken.
        """
        if self.dictate_radio.isChecked():
            return True
        return self.builtin_radio.isChecked() or self._has_voices

    def _refresh_voices(self) -> None:
        from .tts import VoiceCloner

        profiles = VoiceCloner.list_profiles()
        self._has_voices = bool(profiles)
        if profiles and self.cfg.speaker_profile not in profiles:
            self.cfg.speaker_profile = profiles[0]

    def _harvest_settings(self) -> None:
        cfg = self.cfg
        cfg.auto_devices = self.auto_devices_check.isChecked()
        if self._mic:
            cfg.input_device, cfg.input_device_name = self._mic.index, self._mic.name
        sink = self._cable or self._monitor
        if sink:
            cfg.output_device, cfg.output_device_name = sink.index, sink.name
        if self._monitor:
            cfg.monitor_device = self._monitor.index
            cfg.monitor_device_name = self._monitor.name
        cfg.monitor_enabled = (
            self.monitor_check.isChecked()
            and self._monitor is not None
            and sink is not None
            and self._monitor.index != sink.index
        )
        cfg.source_lang = languages.BY_NAME[self.src_combo.currentText()].whisper
        cfg.target_lang = languages.BY_NAME[self.dst_combo.currentText()].whisper
        cfg.speech_quality = self.quality_combo.currentData()
        # The recogniser and translator follow the quality level; they are not
        # chosen separately any more.
        preset = quality.speech(cfg)
        cfg.whisper_model = preset.whisper_model
        cfg.mt_model = preset.mt_model
        cfg.mode = "speak" if self.speak_radio.isChecked() else "dictate"
        cfg.dictate_translate = self.dictate_translate_check.isChecked()
        cfg.voice_mode = "clone" if self.clone_radio.isChecked() else "builtin"
        cfg.builtin_voice = self.builtin_combo.currentData()
        cfg.show_overlay = self.overlay_check.isChecked()
        cfg.save_transcript = self.transcript_check.isChecked()
        if not cfg.duplex_guard_auto:
            cfg.duplex_guard = self.duplex_check.isChecked()
        cfg.save()

    def _ensure_licence(self) -> bool:
        if self.cfg.coqui_tos_agreed:
            return True
        box = QMessageBox(self)
        box.setWindowTitle("XTTS-v2 model licence")
        box.setTextFormat(Qt.RichText)
        box.setText("<b>Before the voice model can be used</b>")
        box.setInformativeText(
            "Speech is made with Coqui XTTS-v2, under the "
            "<a href='https://coqui.ai/cpml'>Coqui Public Model License "
            "1.0.0</a>. Personal and research use only, no commercial use."
            "<br><br>Only clone your own voice, or a voice whose owner agreed."
        )
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        box.button(QMessageBox.Yes).setText("I accept")
        box.button(QMessageBox.No).setText("Decline")
        if box.exec() != QMessageBox.Yes:
            return False
        self.cfg.coqui_tos_agreed = True
        self.cfg.save()
        return True

    # -- quality ---------------------------------------------------------------

    def _refresh_quality_hints(self) -> None:
        preset = quality.BY_KEY.get(self.quality_combo.currentData())
        self.quality_hint.setText(preset.summary if preset else "")

    def _on_quality(self) -> None:
        key = self.quality_combo.currentData()
        self.cfg.speech_quality = key
        preset = quality.BY_KEY.get(key)
        if preset is not None:
            self.cfg.whisper_model = preset.whisper_model
            self.cfg.mt_model = preset.mt_model
        self._refresh_quality_hints()
        if self.pipeline.loaded:
            self._on_status(
                f"Quality set to {preset.name if preset else key}. "
                "Restart Talkbox to apply."
            )

    def _on_duplex(self, on: bool) -> None:
        """Touching this pins it; automatic detection stops overriding."""
        self.cfg.duplex_guard = on
        self.cfg.duplex_guard_auto = False

    # -- actions ---------------------------------------------------------------

    def _swap(self) -> None:
        src, dst = self.src_combo.currentText(), self.dst_combo.currentText()
        self.src_combo.blockSignals(True)
        self.src_combo.setCurrentText(dst)
        self.src_combo.blockSignals(False)
        self.dst_combo.setCurrentText(src)

    def _push_direction(self) -> None:
        src = languages.BY_NAME[self.src_combo.currentText()]
        dst = languages.BY_NAME[self.dst_combo.currentText()]
        self.pipeline.set_direction(
            src.whisper, dst.whisper, self.auto_check.isChecked()
        )

    def _record_voice(self) -> None:
        if self._mic is None:
            QMessageBox.warning(self, "Microphone", "No working input device found.")
            return
        from .tts import VoiceCloner

        dialog = EnrollDialog(self, self._mic, VoiceCloner.list_profiles())
        if dialog.exec() == QDialog.Accepted and dialog.profile_name:
            self.cfg.speaker_profile = dialog.profile_name
            self._refresh_voices()
            if self.pipeline.cloner is not None:
                self.pipeline.cloner.invalidate(dialog.profile_name)
            self._refresh_setup()
            self._on_status(f"Saved voice profile '{dialog.profile_name}'.")

    def _run_wizard(self, first_run: bool = False) -> None:
        from .wizard import SetupWizard

        wizard = SetupWizard(self, self.cfg, STYLE)
        wizard.exec()
        self._rescan()
        if not first_run:
            self._on_status("Setup finished.")

    def _run_selftest(self) -> None:
        if self.pipeline.cloner is None:
            QMessageBox.information(self, "Not ready", "Wait for the models to load.")
            return
        if not self._voice_ready():
            QMessageBox.information(
                self, "Pick a voice first", "The test needs a voice to speak in."
            )
            return
        from . import selftest

        self.selftest_btn.setEnabled(False)
        self.selftest_btn.setText("Testing...")
        # Test whichever voice is actually selected, so the sample the user
        # hears is the one a call would get.
        try:
            if self.builtin_radio.isChecked():
                self.pipeline.cloner.load_builtin(self.builtin_combo.currentData())
            else:
                clone = quality.clone(self.cfg)
                self.pipeline.cloner.load_profile(
                    self.cfg.speaker_profile,
                    gpt_cond_len=clone.gpt_cond_len,
                    max_ref_length=clone.max_ref_length,
                )
        except Exception as exc:
            self._on_selftest_done(False, "Could not load the voice", str(exc))
            return
        selftest.run_async(
            self.pipeline.cloner,
            self._cable,
            languages.get(self.cfg.target_lang).xtts,
            lambda r: self.bridge.selftest.emit(r.ok, r.headline, r.detail),
            progress=self.bridge.status.emit,
        )

    def _on_selftest_done(self, ok: bool, headline: str, detail: str) -> None:
        self.selftest_btn.setEnabled(True)
        self.selftest_btn.setText("Test my setup")
        self._on_status(headline)
        colour = "#3fb950" if ok else "#f0736a"
        self.log.append(
            f'<span style="color:{colour}"><b>{headline}</b></span><br>'
            f'<span style="color:#8b93a3">{detail}</span><br>'
        )
        box = QMessageBox(self)
        box.setWindowTitle("Setup test")
        box.setIcon(QMessageBox.Information if ok else QMessageBox.Warning)
        box.setText(f"<b>{headline}</b>")
        box.setInformativeText(detail)
        box.exec()

    def _toggle(self) -> None:
        if self.pipeline.running:
            self.start_btn.setEnabled(False)
            self._on_status("Stopping...")
            QApplication.processEvents()
            self.pipeline.stop()
            self._set_running(False)
            return
        if not self._voice_ready():
            QMessageBox.information(
                self,
                "Pick a voice first",
                "Nothing has been recorded yet. Either record your voice on "
                "the Voice tab, or choose a built-in voice there instead.",
            )
            return
        self._harvest_settings()
        self._push_direction()
        try:
            self.pipeline.start()
        except Exception as exc:
            QMessageBox.critical(self, "Could not start", str(exc))
            return
        self._set_running(True)

    def _set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Stop" if running else "Start")
        if not running:
            self.vu.setValue(0)
        manual = not self.auto_devices_check.isChecked()
        for widget in (
            self.record_btn,
            self.speak_radio,
            self.dictate_radio,
            self.dictate_translate_check,
            self.clone_radio,
            self.builtin_radio,
            self.builtin_combo,
            self.rescan_btn,
            self.selftest_btn,
            self.wizard_btn,
            self.quality_combo,
            self.transcript_check,
            self.auto_devices_check,
            self.monitor_check,
        ):
            widget.setEnabled(not running)
        for widget in (self.input_combo, self.output_combo):
            widget.setEnabled(not running and manual)
        self.monitor_combo.setEnabled(
            not running and manual and self.monitor_check.isChecked()
        )
        if not running:
            self._update_start_enabled()

    def _toggle_overlay(self, shown: bool) -> None:
        self.cfg.show_overlay = shown
        self.overlay.setVisible(shown)

    # -- signal handlers -------------------------------------------------------

    def _on_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_ready(self) -> None:
        self._models_ready = True
        self._update_start_enabled()
        self._on_status(
            "Ready."
            if self._voice_ready()
            else "Ready. Record your voice, or pick a built-in one."
        )

    def _on_error(self, text: str) -> None:
        self.status_label.setText(text)
        self.log.append(f'<span style="color:#f0736a">{text}</span>')

    def _on_source(self, text: str) -> None:
        self.log.append(f'<span style="color:#8b93a3">{text}</span>')

    def _on_translation(self, text: str) -> None:
        self.log.append(f'<span style="color:#e6e8ec"><b>{text}</b></span><br>')
        self.overlay.set_translation(text)

    def _on_unclear(self, text: str) -> None:
        self.log.append(
            f'<span style="color:#6e7681"><i>unclear, not spoken:</i> {text}</span><br>'
        )
        self.overlay.set_source("(unclear) " + text)

    def _on_dictation(self, text: str, typed: bool) -> None:
        if typed:
            self.log.append(f"{text}<br>")
            return
        # The only reason it is not typed is that this window had focus, and
        # saying so is more useful than dropping the sentence silently.
        self.log.append(
            f'<span style="color:#d29922">Not typed, this window was in '
            f"front:</span> {text}<br>"
        )

    def _on_talkover(self) -> None:
        """The mic gate threw away speech. Say so rather than lose it quietly."""
        self.log.append(
            '<span style="color:#d29922">You spoke while it was talking. '
            "The mic is off then, so that was not translated. Turn that off in "
            "Audio if you are on headphones.</span><br>"
        )
        self._on_status("Spoke over the translation, so that was not sent.")

    def _on_latency(self, seconds: float) -> None:
        self._latency_ms = seconds * 1000
        self._refresh_stats()

    def _on_busy(self, pending: int, speaking: int) -> None:
        self._queues = (pending, speaking)
        self._refresh_stats()

    def _refresh_stats(self) -> None:
        pending, speaking = self._queues
        self.stat_label.setText(
            f"lag {self._latency_ms:5.0f} ms   queue {pending}/{speaking}"
        )

    def _on_level(self, rms: float) -> None:
        self.vu.setValue(int(min(1.0, rms * 6.0) * 100))

    # -- shutdown --------------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802
        try:
            self._harvest_settings()
        except Exception:
            pass
        if self.pipeline.running:
            self.pipeline.stop()
        self.overlay.close()
        super().closeEvent(event)


def run() -> int:
    import sys

    app = QApplication(sys.argv)
    app.setApplicationName("Talkbox")
    cfg = Config.load()
    window = MainWindow(cfg)

    screen = app.primaryScreen().availableGeometry()
    width = min(max(window.sizeHint().width(), 820), screen.width() - 80)
    height = min(max(window.sizeHint().height(), 640), screen.height() - 80)
    window.resize(width, height)
    window.move(
        screen.center().x() - width // 2,
        max(screen.top() + 20, screen.center().y() - height // 2),
    )
    window.show()

    if cfg.show_overlay:
        window.overlay.move(
            screen.center().x() - window.overlay.width() // 2,
            screen.bottom() - window.overlay.height() - 70,
        )
        window.overlay.show()
    return app.exec()
