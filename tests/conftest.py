"""
conftest.py — inject lightweight stubs for heavy ML/GUI deps before any test imports.
This lets unit tests run without torch, PyQt6, ultralytics, insightface or mediapipe.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


# ── torch stub ───────────────────────────────────────────────────────────────
def _stub_torch():
    torch = types.ModuleType("torch")
    torch.cuda = MagicMock()
    torch.cuda.is_available = MagicMock(return_value=False)
    torch.backends = MagicMock()
    torch.backends.cudnn = MagicMock()
    sys.modules.setdefault("torch", torch)

# ── PyQt6 stub ───────────────────────────────────────────────────────────────
def _stub_pyqt6():
    def _mod(name):
        return sys.modules.setdefault(name, types.ModuleType(name))

    pyqt6      = _mod("PyQt6")
    core       = _mod("PyQt6.QtCore")
    gui        = _mod("PyQt6.QtGui")
    widgets    = _mod("PyQt6.QtWidgets")

    # QThread minimal stub
    class _QThread:
        def __init__(self): pass
        def isRunning(self): return False
        def wait(self, ms=0): pass
        def start(self): pass

    # pyqtSignal stub — returns a MagicMock that supports .emit() and .connect()
    def pyqtSignal(*_args, **_kwargs):
        m = MagicMock()
        m.emit   = MagicMock()
        m.connect = MagicMock()
        return m

    core.QThread      = _QThread
    core.pyqtSignal   = pyqtSignal
    core.Qt           = MagicMock()
    gui.QImage        = MagicMock()
    gui.QFont         = MagicMock()
    gui.QPixmap       = MagicMock()

    for name in (
        "QApplication", "QMainWindow", "QWidget", "QLabel", "QPushButton",
        "QComboBox", "QLineEdit", "QTextEdit", "QSlider", "QGroupBox",
        "QVBoxLayout", "QHBoxLayout", "QFormLayout", "QTabWidget",
        "QFileDialog", "QMessageBox", "QStatusBar",
    ):
        setattr(widgets, name, MagicMock())

# ── ultralytics stub ─────────────────────────────────────────────────────────
def _stub_ultralytics():
    stub = types.ModuleType("ultralytics")
    stub.YOLO = MagicMock()
    sys.modules.setdefault("ultralytics", stub)

# ── insightface stub ─────────────────────────────────────────────────────────
def _stub_insightface():
    stub    = types.ModuleType("insightface")
    app_mod = types.ModuleType("insightface.app")
    app_mod.FaceAnalysis = MagicMock()
    stub.app = app_mod
    sys.modules.setdefault("insightface",     stub)
    sys.modules.setdefault("insightface.app", app_mod)

# ── mediapipe stub ───────────────────────────────────────────────────────────
def _stub_mediapipe():
    mp_stub   = types.ModuleType("mediapipe")
    solutions = types.ModuleType("mediapipe.solutions")
    hands_mod = types.ModuleType("mediapipe.solutions.hands")
    draw_mod  = types.ModuleType("mediapipe.solutions.drawing_utils")
    style_mod = types.ModuleType("mediapipe.solutions.drawing_styles")

    hands_mod.Hands            = MagicMock()
    hands_mod.HAND_CONNECTIONS = None
    draw_mod.draw_landmarks    = MagicMock()
    style_mod.get_default_hand_landmarks_style   = MagicMock(return_value=None)
    style_mod.get_default_hand_connections_style = MagicMock(return_value=None)

    mp_stub.solutions = solutions
    solutions.hands          = hands_mod
    solutions.drawing_utils  = draw_mod
    solutions.drawing_styles = style_mod

    sys.modules.setdefault("mediapipe",                          mp_stub)
    sys.modules.setdefault("mediapipe.solutions",                solutions)
    sys.modules.setdefault("mediapipe.solutions.hands",          hands_mod)
    sys.modules.setdefault("mediapipe.solutions.drawing_utils",  draw_mod)
    sys.modules.setdefault("mediapipe.solutions.drawing_styles", style_mod)


# Install all stubs before any test module is imported
_stub_torch()
_stub_pyqt6()
_stub_ultralytics()
_stub_insightface()
_stub_mediapipe()
