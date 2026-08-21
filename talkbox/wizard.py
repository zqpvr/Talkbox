"""First-run setup.

Nothing large is shipped with the source. Models are ~7 GB and the virtual
audio driver belongs to VB-Audio, so both are fetched here on first run
instead. The wizard walks through the steps in the order they can actually be
done, and each one reports what it found rather than assuming.

Steps: system check, licence, model download, virtual cable, record voice,
test. Every step can be skipped, and the wizard can be reopened later from the
Audio tab.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import audio as audio_io
from . import quality
from .config import MODELS_DIR, Config

CABLE_URL = "https://vb-audio.com/Cable/"
CABLE_ZIP = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip"


class Worker(QObject):
    """Runs a slow job off the GUI thread and reports back."""

    progress = Signal(str)
    done = Signal(bool, str)

    def run(self, fn) -> None:
        def body() -> None:
            try:
                message = fn(self.progress.emit)
                self.done.emit(True, message or "Done.")
            except Exception as exc:
                self.done.emit(False, f"{type(exc).__name__}: {exc}")

        threading.Thread(target=body, daemon=True).start()


# --------------------------------------------------------------------------- #
#  Work performed by the steps
# --------------------------------------------------------------------------- #


def system_report() -> tuple[bool, list[str]]:
    """What the machine can do. Returns (usable, lines)."""
    lines: list[str] = []
    ok = True
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = torch.cuda.mem_get_info()[0] / 1024**3
            lines.append(f"GPU: {name}")
            lines.append(f"VRAM: {total:.1f} GB total, {free:.1f} GB free now")
            if total < 5.5:
                lines.append("Only the Low preset will fit on this card.")
            if free < total - 2.0:
                lines.append(
                    "Something else is using the GPU. Close it before a call, "
                    "or expect slower speech."
                )
        else:
            ok = False
            lines.append("No NVIDIA GPU found. Dictation will still work on")
            lines.append("the CPU. Speaking into a call will not: the voice")
            lines.append("model runs slower than real time there.")
    except Exception as exc:
        ok = False
        lines.append(f"PyTorch is not working: {exc}")

    missing = []
    for module, label in (
        ("faster_whisper", "faster-whisper"),
        ("TTS", "coqui-tts"),
        ("silero_vad", "silero-vad"),
        ("sounddevice", "sounddevice"),
    ):
        try:
            __import__(module)
        except Exception:
            missing.append(label)
    if missing:
        ok = False
        lines.append("Missing packages: " + ", ".join(missing))
        lines.append("Run setup.ps1 again.")
    else:
        lines.append("All required packages are installed.")

    mic = audio_io.autoselect_input()
    lines.append(f"Microphone: {mic.short}" if mic else "No working microphone found.")
    if mic is None:
        ok = False
    return ok, lines


def download_models(cfg: Config, report) -> str:
    """Fetch everything the chosen quality level needs.

    Roughly 3 GB the first time. Nothing is bundled with the source, so this is
    where it arrives.
    """
    preset = quality.speech(cfg)

    report(f"Downloading speech recogniser ({preset.whisper_model})...")
    from .asr import Transcriber

    Transcriber(preset.whisper_model, "cpu", "int8")

    report("Downloading translator...")
    from .mt import convert, is_converted

    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(preset.mt_model, cache_dir=str(MODELS_DIR / "hf"))
    if not is_converted(preset.mt_model):
        report("Converting translator for speed (about two minutes, one time)...")
        convert(preset.mt_model, report)

    report("Downloading voice model (about 1.9 GB)...")
    from .tts import download_model

    download_model(report)
    return "All models downloaded."


def fetch_cable(report) -> str:
    """Download and unpack VB-Cable. Installing it needs admin, so that part
    is left to the user."""
    import io
    import urllib.request
    import zipfile

    target = Path(__file__).resolve().parent.parent / "vbcable"
    exe = target / "VBCABLE_Setup_x64.exe"
    if exe.exists():
        return f"Already downloaded to {target}"

    report("Downloading VB-Cable from vb-audio.com...")
    with urllib.request.urlopen(CABLE_ZIP, timeout=120) as response:
        blob = response.read()
    report("Unpacking...")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        archive.extractall(target)
    if not exe.exists():
        raise RuntimeError("The download did not contain the expected installer.")
    return str(target)


# --------------------------------------------------------------------------- #
#  The wizard
# --------------------------------------------------------------------------- #


class Step(QWidget):
    def __init__(self, title: str, body: str):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)

        heading = QLabel(title)
        font = QFont()
        font.setPointSize(13)
        font.setWeight(QFont.DemiBold)
        heading.setFont(font)
        self.layout.addWidget(heading)

        self.body = QLabel(body)
        self.body.setWordWrap(True)
        self.body.setStyleSheet("color: #8b93a3;")
        self.layout.addWidget(self.body)

        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color: #e6e8ec;")
        self.layout.addWidget(self.detail)

        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(False)
        self.bar.hide()
        self.layout.addWidget(self.bar)

        self.action = QPushButton("")
        self.action.hide()
        self.layout.addWidget(self.action, 0, Qt.AlignLeft)
        self.layout.addStretch(1)

    def busy(self, on: bool) -> None:
        self.bar.setVisible(on)
        self.action.setEnabled(not on)


class SetupWizard(QDialog):
    def __init__(self, parent, cfg: Config, style: str = ""):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("Talkbox setup")
        self.setMinimumSize(640, 460)
        if style:
            self.setStyleSheet(style)

        self.completed = False
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        self.step_label = QLabel("")
        self.step_label.setStyleSheet("color: #8b93a3;")
        self.back_btn = QPushButton("Back")
        self.next_btn = QPushButton("Next")
        self.next_btn.setObjectName("primary")
        self.skip_btn = QPushButton("Skip")
        nav = QHBoxLayout()
        nav.addWidget(self.step_label, 1)
        nav.addWidget(self.skip_btn)
        nav.addWidget(self.back_btn)
        nav.addWidget(self.next_btn)
        root.addLayout(nav)

        self.back_btn.clicked.connect(lambda: self.goto(self.stack.currentIndex() - 1))
        self.next_btn.clicked.connect(lambda: self.goto(self.stack.currentIndex() + 1))
        self.skip_btn.clicked.connect(lambda: self.goto(self.stack.currentIndex() + 1))

        self._build()
        self.goto(0)

    # -- steps ---------------------------------------------------------------

    def _build(self) -> None:
        self.s_check = Step(
            "Checking your machine",
            "Talkbox needs an NVIDIA GPU to keep up with a live call.",
        )
        self.stack.addWidget(self.s_check)

        self.s_models = Step(
            "Download the models",
            "About 3 GB. Nothing is bundled with the app, so this happens once. "
            "You can carry on using the rest of the wizard while it runs.",
        )
        self.s_models.action.setText("Download now")
        self.s_models.action.show()
        self.s_models.action.clicked.connect(self._do_models)
        self.stack.addWidget(self.s_models)

        self.s_cable = Step(
            "Let calls hear you",
            "Windows cannot feed audio to Discord or Zoom as a microphone "
            "without a virtual audio driver. VB-Cable is free. Talkbox can "
            "download it, but installing a driver needs administrator rights, "
            "so you have to run the installer yourself and reboot.",
        )
        self.s_cable.action.setText("Download VB-Cable")
        self.s_cable.action.show()
        self.s_cable.action.clicked.connect(self._do_cable)
        self.stack.addWidget(self.s_cable)

        from .enroll import CLIP_PROMPTS

        self.s_voice = Step(
            "Record your voice",
            f"{len(CLIP_PROMPTS)} short clips, about a minute in total. This is "
            "the only description of your voice the app gets, so record "
            "somewhere quiet. You can skip this and use one of the voices "
            "built into the model instead.",
        )
        self.s_voice.action.setText("Record now")
        self.s_voice.action.show()
        self.s_voice.action.clicked.connect(self._do_voice)
        self.s_voice_skip = QPushButton("Use a built-in voice instead")
        self.s_voice_skip.clicked.connect(self._skip_voice)
        # Step ends its layout with a stretch, so append before that or the
        # button is pushed to the bottom of the page away from the other one.
        self.s_voice.layout.insertWidget(
            self.s_voice.layout.count() - 1, self.s_voice_skip, 0, Qt.AlignLeft
        )
        self.stack.addWidget(self.s_voice)

        self.s_done = Step(
            "Ready",
            "Set your call application's microphone to the virtual cable, pick "
            "your languages, and press Start.",
        )
        self.stack.addWidget(self.s_done)

    def goto(self, index: int) -> None:
        index = max(0, min(self.stack.count() - 1, index))
        self.stack.setCurrentIndex(index)
        self.step_label.setText(f"Step {index + 1} of {self.stack.count()}")
        self.back_btn.setEnabled(index > 0)
        last = index == self.stack.count() - 1
        self.next_btn.setText("Finish" if last else "Next")
        self.skip_btn.setVisible(not last)
        try:
            self.next_btn.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        if last:
            self.next_btn.clicked.connect(self._finish)
        else:
            self.next_btn.clicked.connect(
                lambda: self.goto(self.stack.currentIndex() + 1)
            )
        self._refresh(index)

    def _refresh(self, index: int) -> None:
        if index == 0:
            ok, lines = system_report()
            self.s_check.detail.setText("\n".join(lines))
        elif index == 1:
            self.s_models.detail.setText(self._models_state())
        elif index == 2:
            cable = audio_io.autoselect_call_sink()
            if cable is not None:
                self.s_cable.detail.setText(
                    f"Installed: {cable.short}\n"
                    f'In your call app, choose "{audio_io.call_source_name(cable)}" '
                    f"as the microphone."
                )
                self.s_cable.action.setText("Download again")
            else:
                exe = (
                    Path(__file__).resolve().parent.parent
                    / "vbcable"
                    / "VBCABLE_Setup_x64.exe"
                )
                if exe.exists():
                    self.s_cable.detail.setText(
                        f"Downloaded to:\n{exe}\n\n"
                        "Right-click it, Run as administrator, then reboot."
                    )
                    self.s_cable.action.setText("Open the folder")
                else:
                    self.s_cable.detail.setText("Not installed yet.")
        elif index == 3:
            from .tts import VoiceCloner

            profiles = VoiceCloner.list_profiles()
            if self.cfg.voice_mode == "builtin":
                self.s_voice.detail.setText(
                    "Using the built-in voice %s. Record any time from the "
                    "Voice tab to switch to your own." % self.cfg.builtin_voice
                )
            else:
                self.s_voice.detail.setText(
                    "Recorded: " + ", ".join(profiles)
                    if profiles
                    else "Not recorded yet."
                )
        elif index == 4:
            self.s_done.detail.setText(self._summary())

    def _models_state(self) -> str:
        from .mt import is_converted

        preset = quality.speech(self.cfg)
        whisper_dir = MODELS_DIR / "whisper"
        have_whisper = whisper_dir.is_dir() and any(whisper_dir.rglob("model.bin"))
        hf = MODELS_DIR / "hf"
        have_xtts = hf.is_dir() and any(hf.rglob("model.pth"))
        rows = [
            ("Speech recogniser", have_whisper),
            ("Translator", is_converted(preset.mt_model)),
            ("Voice model", have_xtts),
        ]
        return "\n".join(
            f"{'ready' if ok else 'not downloaded'}   {name}" for name, ok in rows
        )

    def _summary(self) -> str:
        from .tts import VoiceCloner

        mic = audio_io.autoselect_input()
        cable = audio_io.autoselect_call_sink()
        voices = VoiceCloner.list_profiles()
        return "\n".join(
            [
                f"Microphone   {mic.short if mic else 'none found'}",
                f"Call hears   {cable.short if cable else 'no virtual cable yet'}",
                f"Your voice   {', '.join(voices) if voices else 'not recorded'}",
            ]
        )

    # -- actions -------------------------------------------------------------

    def _run(self, step: Step, fn) -> None:
        step.busy(True)
        worker = Worker(self)
        worker.progress.connect(step.detail.setText)

        def finished(ok: bool, message: str) -> None:
            step.busy(False)
            step.detail.setText(message)
            self._refresh(self.stack.currentIndex())

        worker.done.connect(finished)
        self._worker = worker  # keep a reference alive
        worker.run(fn)

    def _do_models(self) -> None:
        self._run(self.s_models, lambda report: download_models(self.cfg, report))

    def _do_cable(self) -> None:
        cable = audio_io.autoselect_call_sink()
        folder = Path(__file__).resolve().parent.parent / "vbcable"
        if cable is not None or (folder / "VBCABLE_Setup_x64.exe").exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
            return
        self._run(self.s_cable, fetch_cable)

    def _do_voice(self) -> None:
        from .tts import VoiceCloner
        from .ui import EnrollDialog

        mic = audio_io.autoselect_input()
        if mic is None:
            self.s_voice.detail.setText("No working microphone found.")
            return
        dialog = EnrollDialog(self, mic, VoiceCloner.list_profiles())
        if dialog.exec() and dialog.profile_name:
            self.cfg.speaker_profile = dialog.profile_name
            self.cfg.voice_mode = "clone"
            self.cfg.save()
        self._refresh(self.stack.currentIndex())

    def _skip_voice(self) -> None:
        """Finish setup with no recording at all.

        The built-in speakers live in the XTTS checkpoint that was downloaded
        two steps ago, so there is nothing further to fetch and nothing to
        record. Everything except the voice matching the user works.
        """
        self.cfg.voice_mode = "builtin"
        self.cfg.save()
        self._refresh(self.stack.currentIndex())

    def _finish(self) -> None:
        self.completed = True
        self.cfg.setup_complete = True
        self.cfg.save()
        self.accept()


def needs_setup(cfg: Config) -> bool:
    """First run, or the essentials are missing."""
    if not cfg.setup_complete:
        return True
    if cfg.voice_mode == "builtin":
        # Nothing to record, so nothing can be missing.
        return False
    from .tts import VoiceCloner

    return not VoiceCloner.list_profiles()
