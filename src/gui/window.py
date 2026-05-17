"""Main application window — Object Detection, Face Recognition, Gesture Recognition."""
from __future__ import annotations

import json
import pathlib
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch
from loguru import logger
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.engine import DetectionEngine
from src.core.face_engine import FaceRecognitionEngine
from src.core.gesture_engine import GestureRecognitionEngine
from src.utils.helpers import ensure_dir

BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

_MAX_HISTORY = 5_000   # cap per-session detection lists to avoid unbounded RAM


# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    """Three-tab GUI for detection, face-recognition and gesture engines."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Detector + FaceRec + Gestures v2.2")
        self.setMinimumSize(1280, 860)

        ensure_dir(MODELS_DIR)
        ensure_dir(OUTPUT_DIR)

        # engine handles
        self.det_engine:     Optional[DetectionEngine]       = None
        self.face_engine:    Optional[FaceRecognitionEngine] = None
        self.gesture_engine: Optional[GestureRecognitionEngine] = None

        # session state
        self.source_path:   str = ""
        self.active_module: str = ""
        self.detection_history: List[Dict] = []
        self.face_history:      List[Dict] = []
        self.gesture_history:   List[Dict] = []
        self._last_ui_update:   float = 0.0

        # config snapshots (updated from UI on START)
        _default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.det_config: Dict = {
            "source_type": "webcam", "source_path": "",
            "model_name": "yolo26s.pt",
            "device": _default_device,
            "conf": 0.25, "iou": 0.45, "imgsz": 640,
        }
        self.face_config: Dict = {
            "source_type": "webcam", "source_path": "",
            "device": "cpu",
            "det_thresh": 0.5, "rec_thresh": 0.35,
            "face_db_dir": "", "imgsz": 320, "skip_frames": 3,
        }
        self.gesture_config: Dict = {
            "source_type": "webcam", "source_path": "",
            "device": "cpu",
            "min_detection_confidence": 0.5,
            "min_tracking_confidence":  0.5,
            "max_num_hands": 2,
        }

        self._build_ui()
        logger.info("🚀 Application started")

    # ══════════════════════════════════════════════════════════ UI construction
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_detection(),  "🔍 Object Detection")
        self.tabs.addTab(self._tab_face(),        "👤 Face Recognition")
        self.tabs.addTab(self._tab_gesture(),     "🖐️ Gesture Recognition")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self.tabs)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            "color: #0f0; background: #111; font-weight: 500;"
        )
        self.status_bar.showMessage("✅ Ready — pick a tab, configure and click START")
        self.setStatusBar(self.status_bar)

    # ─────────────── shared helpers ──────────────────────────────────────────
    @staticmethod
    def _group(title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet(
            "QGroupBox { font-weight: 600; color: #fff; border: 1px solid #444;"
            " border-radius: 8px; margin-top: 8px; padding-top: 12px;"
            " background: #1e1e1e; }"
            " QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }"
        )
        return g

    @staticmethod
    def _btn(label: str, color: str) -> QPushButton:
        b = QPushButton(label)
        b.setStyleSheet(
            f"QPushButton {{ background:{color}; color:#fff; border:none;"
            f" border-radius:6px; padding:8px 16px; font-weight:500; }}"
            f" QPushButton:hover {{ border:1px solid rgba(255,255,255,0.3); }}"
            f" QPushButton:disabled {{ background:#444; color:#888; }}"
        )
        return b

    @staticmethod
    def _slider_row(slider: QSlider, label: QLabel) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(slider)
        lay.addWidget(label)
        return w

    @staticmethod
    def _h_slider(lo: int, hi: int, val: int) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    # ══════════════════════════════════════════════════════════ Detection tab
    def _tab_detection(self) -> QWidget:
        tab = QWidget()
        lay = QHBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8)

        # ── left: video area
        left = QVBoxLayout()
        self.det_video = QLabel("Select source and click START")
        self.det_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.det_video.setStyleSheet(
            "background:#0a0a0a; color:#555; border:2px solid #333;"
            " border-radius:10px; min-height:460px; font-size:14px;"
        )
        self.det_stat_lbl = QLabel("Idle")
        self.det_stat_lbl.setStyleSheet("color:#888; font-size:13px;")
        left.addWidget(self.det_video)
        left.addWidget(self.det_stat_lbl)
        left_w = QWidget(); left_w.setLayout(left)
        lay.addWidget(left_w, 3)

        # ── right: controls
        right = QVBoxLayout()
        right.setSpacing(10)

        # source
        g = self._group("📁 Source"); fl = QVBoxLayout(g)
        self.det_src = QComboBox()
        self.det_src.addItems(["📹 Webcam", "🎬 Video File", "🖼️ Image"])
        self.det_src.currentIndexChanged.connect(self._det_src_changed)
        self.det_file_btn = QPushButton("📂 Select File…")
        self.det_file_btn.hide()
        self.det_file_btn.clicked.connect(self._det_pick_file)
        fl.addWidget(self.det_src); fl.addWidget(self.det_file_btn)
        right.addWidget(g)

        # model
        g = self._group("🤖 Model & Device"); fl = QFormLayout(g)
        self.det_model = QComboBox()
        self.det_model.addItems(["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt"])
        self.det_model.setCurrentText("yolo26s.pt")
        self.det_device = QComboBox()
        self.det_device.addItems(["CPU", "CUDA"])
        if torch.cuda.is_available():
            self.det_device.setCurrentIndex(1)
        fl.addRow("Model:", self.det_model)
        fl.addRow("Device:", self.det_device)
        right.addWidget(g)

        # thresholds
        g = self._group("⚙️ Thresholds"); fl = QFormLayout(g)
        self.det_sld_conf = self._h_slider(5, 95, 25)
        self.det_lbl_conf = QLabel("0.25")
        self.det_sld_conf.valueChanged.connect(
            lambda v: self.det_lbl_conf.setText(f"{v/100:.2f}")
        )
        self.det_sld_iou = self._h_slider(5, 95, 45)
        self.det_lbl_iou = QLabel("0.45")
        self.det_sld_iou.valueChanged.connect(
            lambda v: self.det_lbl_iou.setText(f"{v/100:.2f}")
        )
        self.det_imgsz = QLineEdit("640")
        fl.addRow("Confidence:", self._slider_row(self.det_sld_conf, self.det_lbl_conf))
        fl.addRow("IoU (NMS):", self._slider_row(self.det_sld_iou, self.det_lbl_iou))
        fl.addRow("Input size:", self.det_imgsz)
        right.addWidget(g)

        # control
        g = self._group("🎮 Control"); fl = QVBoxLayout(g)
        br = QHBoxLayout()
        self.det_start = self._btn("▶ START", "#2ecc71")
        self.det_start.clicked.connect(self._det_start)
        self.det_stop = self._btn("⏹ STOP", "#e74c3c")
        self.det_stop.setEnabled(False)
        self.det_stop.clicked.connect(self._det_stop)
        br.addWidget(self.det_start); br.addWidget(self.det_stop)
        fl.addLayout(br)
        right.addWidget(g)

        # export
        g = self._group("💾 Export"); fl = QHBoxLayout(g)
        self.det_rec  = self._btn("🔴 REC",   "#555")
        self.det_shot = self._btn("📸 SC",    "#555")
        self.det_json = self._btn("📝 JSON",  "#555")
        self.det_rec.clicked.connect(self._det_toggle_rec)
        self.det_shot.clicked.connect(self._det_screenshot)
        self.det_json.clicked.connect(self._det_save_json)
        fl.addWidget(self.det_rec); fl.addWidget(self.det_shot); fl.addWidget(self.det_json)
        right.addWidget(g)

        # results
        g = self._group("🎯 Detected Objects"); fl = QVBoxLayout(g)
        self.det_results = QTextEdit()
        self.det_results.setReadOnly(True)
        self.det_results.setStyleSheet(
            "background:#111; color:#0f0; font-family:Consolas; font-size:11px;"
        )
        fl.addWidget(self.det_results)
        right.addWidget(g)
        right.addStretch()

        right_w = QWidget(); right_w.setLayout(right)
        lay.addWidget(right_w, 1)
        return tab

    # ══════════════════════════════════════════════════════════ Face tab
    def _tab_face(self) -> QWidget:
        tab = QWidget()
        lay = QHBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8)

        left = QVBoxLayout()
        self.face_video = QLabel("Select source and click START")
        self.face_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.face_video.setStyleSheet(
            "background:#0a0a0a; color:#555; border:2px solid #333;"
            " border-radius:10px; min-height:460px; font-size:14px;"
        )
        self.face_stat_lbl = QLabel("Idle")
        self.face_stat_lbl.setStyleSheet("color:#888; font-size:13px;")
        left.addWidget(self.face_video)
        left.addWidget(self.face_stat_lbl)
        left_w = QWidget(); left_w.setLayout(left)
        lay.addWidget(left_w, 3)

        right = QVBoxLayout()
        right.setSpacing(10)

        # source
        g = self._group("📁 Source"); fl = QVBoxLayout(g)
        self.face_src = QComboBox()
        self.face_src.addItems(["📹 Webcam", "🎬 Video File", "🖼️ Image"])
        self.face_src.currentIndexChanged.connect(self._face_src_changed)
        self.face_file_btn = QPushButton("📂 Select File…")
        self.face_file_btn.hide()
        self.face_file_btn.clicked.connect(self._face_pick_file)
        fl.addWidget(self.face_src); fl.addWidget(self.face_file_btn)
        right.addWidget(g)

        # face DB
        g = self._group("👥 Face Database"); fl = QVBoxLayout(g)
        self.face_db_btn = QPushButton("📂 Load DB Folder…")
        self.face_db_btn.clicked.connect(self._face_load_db)
        self.face_db_lbl = QLabel("No DB loaded — all faces → Unknown")
        self.face_db_lbl.setStyleSheet("color:#aaa; font-size:12px;")
        fl.addWidget(self.face_db_btn); fl.addWidget(self.face_db_lbl)
        right.addWidget(g)

        # thresholds
        g = self._group("⚙️ Thresholds"); fl = QFormLayout(g)
        self.face_sld_det = self._h_slider(10, 90, 50)
        self.face_lbl_det = QLabel("0.50")
        self.face_sld_det.valueChanged.connect(
            lambda v: self.face_lbl_det.setText(f"{v/100:.2f}")
        )
        self.face_sld_rec = self._h_slider(10, 80, 35)
        self.face_lbl_rec = QLabel("0.35")
        self.face_sld_rec.valueChanged.connect(
            lambda v: self.face_lbl_rec.setText(f"{v/100:.2f}")
        )
        fl.addRow("Det conf:",   self._slider_row(self.face_sld_det, self.face_lbl_det))
        fl.addRow("Rec sim:",    self._slider_row(self.face_sld_rec, self.face_lbl_rec))
        right.addWidget(g)

        # control
        g = self._group("🎮 Control"); fl = QVBoxLayout(g)
        br = QHBoxLayout()
        self.face_start = self._btn("▶ START", "#3498db")
        self.face_start.clicked.connect(self._face_start)
        self.face_stop = self._btn("⏹ STOP", "#e74c3c")
        self.face_stop.setEnabled(False)
        self.face_stop.clicked.connect(self._face_stop)
        br.addWidget(self.face_start); br.addWidget(self.face_stop)
        fl.addLayout(br)
        right.addWidget(g)

        # export
        g = self._group("💾 Export"); fl = QHBoxLayout(g)
        self.face_rec  = self._btn("🔴 REC",  "#555")
        self.face_shot = self._btn("📸 SC",   "#555")
        self.face_json = self._btn("📝 JSON", "#555")
        self.face_rec.clicked.connect(self._face_toggle_rec)
        self.face_shot.clicked.connect(self._face_screenshot)
        self.face_json.clicked.connect(self._face_save_json)
        fl.addWidget(self.face_rec); fl.addWidget(self.face_shot); fl.addWidget(self.face_json)
        right.addWidget(g)

        # results
        g = self._group("👤 Recognized Faces"); fl = QVBoxLayout(g)
        self.face_results = QTextEdit()
        self.face_results.setReadOnly(True)
        self.face_results.setStyleSheet(
            "background:#111; color:#0af; font-family:Consolas; font-size:11px;"
        )
        fl.addWidget(self.face_results)
        right.addWidget(g)
        right.addStretch()

        right_w = QWidget(); right_w.setLayout(right)
        lay.addWidget(right_w, 1)
        return tab

    # ══════════════════════════════════════════════════════════ Gesture tab
    def _tab_gesture(self) -> QWidget:
        tab = QWidget()
        lay = QHBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8)

        left = QVBoxLayout()
        self.gesture_video = QLabel("Select source and click START")
        self.gesture_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gesture_video.setStyleSheet(
            "background:#0a0a0a; color:#555; border:2px solid #333;"
            " border-radius:10px; min-height:460px; font-size:14px;"
        )
        self.gesture_stat_lbl = QLabel("Idle")
        self.gesture_stat_lbl.setStyleSheet("color:#888; font-size:13px;")
        left.addWidget(self.gesture_video)
        left.addWidget(self.gesture_stat_lbl)
        left_w = QWidget(); left_w.setLayout(left)
        lay.addWidget(left_w, 3)

        right = QVBoxLayout()
        right.setSpacing(10)

        # source
        g = self._group("📁 Source"); fl = QVBoxLayout(g)
        self.gesture_src = QComboBox()
        self.gesture_src.addItems(["📹 Webcam", "🎬 Video File", "🖼️ Image"])
        self.gesture_src.currentIndexChanged.connect(self._gesture_src_changed)
        self.gesture_file_btn = QPushButton("📂 Select File…")
        self.gesture_file_btn.hide()
        self.gesture_file_btn.clicked.connect(self._gesture_pick_file)
        fl.addWidget(self.gesture_src); fl.addWidget(self.gesture_file_btn)
        right.addWidget(g)

        # settings
        g = self._group("⚙️ MediaPipe Settings"); fl = QFormLayout(g)
        self.gesture_sld_det = self._h_slider(10, 90, 50)
        self.gesture_lbl_det = QLabel("0.50")
        self.gesture_sld_det.valueChanged.connect(
            lambda v: self.gesture_lbl_det.setText(f"{v/100:.2f}")
        )
        self.gesture_sld_track = self._h_slider(10, 90, 50)
        self.gesture_lbl_track = QLabel("0.50")
        self.gesture_sld_track.valueChanged.connect(
            lambda v: self.gesture_lbl_track.setText(f"{v/100:.2f}")
        )
        self.gesture_hands = QLineEdit("2")
        fl.addRow("Det conf:",   self._slider_row(self.gesture_sld_det,   self.gesture_lbl_det))
        fl.addRow("Track conf:", self._slider_row(self.gesture_sld_track, self.gesture_lbl_track))
        fl.addRow("Max hands:",  self.gesture_hands)
        right.addWidget(g)

        # control
        g = self._group("🎮 Control"); fl = QVBoxLayout(g)
        br = QHBoxLayout()
        self.gesture_start = self._btn("▶ START", "#9b59b6")
        self.gesture_start.clicked.connect(self._gesture_start)
        self.gesture_stop = self._btn("⏹ STOP", "#e74c3c")
        self.gesture_stop.setEnabled(False)
        self.gesture_stop.clicked.connect(self._gesture_stop)
        br.addWidget(self.gesture_start); br.addWidget(self.gesture_stop)
        fl.addLayout(br)
        right.addWidget(g)

        # export
        g = self._group("💾 Export"); fl = QHBoxLayout(g)
        self.gesture_rec  = self._btn("🔴 REC",  "#555")
        self.gesture_shot = self._btn("📸 SC",   "#555")
        self.gesture_json = self._btn("📝 JSON", "#555")
        self.gesture_rec.clicked.connect(self._gesture_toggle_rec)
        self.gesture_shot.clicked.connect(self._gesture_screenshot)
        self.gesture_json.clicked.connect(self._gesture_save_json)
        fl.addWidget(self.gesture_rec); fl.addWidget(self.gesture_shot); fl.addWidget(self.gesture_json)
        right.addWidget(g)

        # results
        g = self._group("🖐️ Recognized Gestures"); fl = QVBoxLayout(g)
        self.gesture_results = QTextEdit()
        self.gesture_results.setReadOnly(True)
        self.gesture_results.setStyleSheet(
            "background:#111; color:#d4a5ff; font-family:Consolas; font-size:11px;"
        )
        fl.addWidget(self.gesture_results)
        right.addWidget(g)
        right.addStretch()

        right_w = QWidget(); right_w.setLayout(right)
        lay.addWidget(right_w, 1)
        return tab

    # ══════════════════════════════════════════════════════════ Tab switching
    def _on_tab_changed(self, _idx: int) -> None:
        self._stop_all()
        self.status_bar.showMessage("✅ Tab switched — Ready")

    def _stop_all(self) -> None:
        if self.det_engine     and self.det_engine.isRunning():     self._det_stop()
        if self.face_engine    and self.face_engine.isRunning():    self._face_stop()
        if self.gesture_engine and self.gesture_engine.isRunning(): self._gesture_stop()

    # ══════════════════════════════════════════════════════════ Detection logic
    def _det_src_changed(self, idx: int) -> None:
        self.det_file_btn.setVisible(idx > 0)
        if idx == 0:
            self.source_path = ""
            self.det_stat_lbl.setText("📹 Webcam selected")

    def _det_pick_file(self) -> None:
        idx = self.det_src.currentIndex()
        ext = ["", "Videos (*.mp4 *.avi *.mkv *.mov)", "Images (*.jpg *.jpeg *.png *.bmp)"][idx]
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", ext)
        if path:
            self.source_path = path
            self.det_stat_lbl.setText(f"📄 {pathlib.Path(path).name}")

    def _det_start(self) -> None:
        if self.active_module == "det":
            self._det_stop()
        try:
            src_idx = self.det_src.currentIndex()
            source_type = ["webcam", "video", "image"][src_idx]
            self.det_config.update({
                "source_type": source_type,
                "source_path": self.source_path if src_idx > 0 else "",
                "model_name":  self.det_model.currentText(),
                "device":      self.det_device.currentText().lower(),
                "conf":        self.det_sld_conf.value() / 100.0,
                "iou":         self.det_sld_iou.value()  / 100.0,
                "imgsz":       int(self.det_imgsz.text() or "640"),
            })
            self.detection_history.clear()
            self.det_results.clear()
            self.det_engine = DetectionEngine(**self.det_config)
            self.det_engine.frame_ready.connect(self._det_frame)
            self.det_engine.stats_ready.connect(self._det_stats)
            self.det_engine.data_ready.connect(self._det_data)
            self.det_engine.error_occurred.connect(self._det_error)
            self.det_engine.finished.connect(self._det_finished)
            self.det_engine.start()
            self.active_module = "det"
            self.det_start.setEnabled(False)
            self.det_stop.setEnabled(True)
            self.det_stat_lbl.setText("⚡ Running…")
            self.status_bar.showMessage("🔥 Detection running")
        except Exception as exc:
            logger.error(f"❌ Det start: {exc}")
            QMessageBox.critical(self, "Error", str(exc))
            self._det_stop()

    def _det_stop(self) -> None:
        if self.det_engine and self.det_engine.isRunning():
            self.det_engine.stop()
        self.det_start.setEnabled(True)
        self.det_stop.setEnabled(False)
        self.det_stat_lbl.setText("⏹️ Stopped")
        self.det_rec.setText("🔴 REC")
        self.det_rec.setStyleSheet(self._btn("🔴 REC", "#555").styleSheet())
        if self.active_module == "det":
            self.active_module = ""
        self.status_bar.showMessage("✅ Detection stopped")

    def _det_finished(self) -> None:
        self.det_start.setEnabled(True)
        self.det_stop.setEnabled(False)
        if self.det_stat_lbl.text() == "⚡ Running…":
            self.det_stat_lbl.setText("✅ Finished")
        if self.active_module == "det":
            self.active_module = ""

    def _det_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Detection Error", msg)
        self._det_stop()

    def _det_toggle_rec(self) -> None:
        if not (self.det_engine and self.det_engine.isRunning()):
            return
        stopped = self.det_engine.toggle_recording()
        if stopped:
            # recording just stopped → button back to "REC"
            self.det_rec.setText("🔴 REC")
            self.status_bar.showMessage(f"💾 Video saved → {OUTPUT_DIR}")
        else:
            # recording just started
            self.det_rec.setText("⏹ STOP REC")
            self.status_bar.showMessage("🔴 Recording…")

    def _det_screenshot(self) -> None:
        pix = self.det_video.pixmap()
        if pix and not pix.isNull():
            path = OUTPUT_DIR / f"det_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            if pix.save(str(path)):
                self.status_bar.showMessage(f"📸 Saved: {path.name}")
            else:
                QMessageBox.critical(self, "Error", "Screenshot failed")

    def _det_save_json(self) -> None:
        if not self.detection_history:
            QMessageBox.warning(self, "No Data", "No detections to export yet")
            return
        self._save_json(self.detection_history, "detection_report")

    def _det_frame(self, img) -> None:
        self.det_video.setPixmap(
            img_to_pixmap(img, self.det_video.size())
        )

    def _det_stats(self, fps: float, lat: float, cnt: int) -> None:
        self.det_stat_lbl.setText(f"FPS: {fps:.1f} | Lat: {lat:.1f}ms | Obj: {cnt}")

    def _det_data(self, data: list) -> None:
        self._extend_capped(self.detection_history, data)
        if not self._throttle_ui():
            return
        if data:
            lines = [
                f"• {d.get('class_name','?')} "
                f"({d.get('confidence', 0):.0%}) "
                f"[{d.get('bbox_xywh', [])}]"
                for d in data[:10]
            ]
            self.det_results.setText("\n".join(lines))

    # ══════════════════════════════════════════════════════════ Face logic
    def _face_src_changed(self, idx: int) -> None:
        self.face_file_btn.setVisible(idx > 0)
        if idx == 0:
            self.face_config["source_path"] = ""
            self.face_stat_lbl.setText("📹 Webcam selected")

    def _face_pick_file(self) -> None:
        idx = self.face_src.currentIndex()
        ext = ["", "Videos (*.mp4 *.avi *.mkv *.mov)", "Images (*.jpg *.jpeg *.png *.bmp)"][idx]
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", ext)
        if path:
            self.face_config["source_path"] = path
            self.face_stat_lbl.setText(f"📄 {pathlib.Path(path).name}")

    def _face_load_db(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Face DB folder")
        if folder:
            self.face_config["face_db_dir"] = folder
            count = len([
                p for p in pathlib.Path(folder).iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ])
            self.face_db_lbl.setText(
                f"✅ DB: {pathlib.Path(folder).name} ({count} image(s))"
            )

    def _face_start(self) -> None:
        if self.active_module == "face":
            self._face_stop()
        try:
            src_idx = self.face_src.currentIndex()
            self.face_config.update({
                "source_type": ["webcam", "video", "image"][src_idx],
                "det_thresh":  self.face_sld_det.value() / 100.0,
                "rec_thresh":  self.face_sld_rec.value() / 100.0,
                "imgsz":       320,
                "skip_frames": 3,
            })
            self.face_history.clear()
            self.face_results.clear()
            self.face_engine = FaceRecognitionEngine(**self.face_config)
            self.face_engine.frame_ready.connect(self._face_frame)
            self.face_engine.stats_ready.connect(self._face_stats)
            self.face_engine.data_ready.connect(self._face_data)
            self.face_engine.error_occurred.connect(self._face_error)
            self.face_engine.finished.connect(self._face_finished)
            self.face_engine.start()
            self.active_module = "face"
            self.face_start.setEnabled(False)
            self.face_stop.setEnabled(True)
            self.face_stat_lbl.setText("⚡ Running…")
            self.status_bar.showMessage("🔥 Face Recognition running")
        except Exception as exc:
            logger.error(f"❌ Face start: {exc}")
            QMessageBox.critical(self, "Error", str(exc))
            self._face_stop()

    def _face_stop(self) -> None:
        if self.face_engine and self.face_engine.isRunning():
            self.face_engine.stop()
        self.face_start.setEnabled(True)
        self.face_stop.setEnabled(False)
        self.face_stat_lbl.setText("⏹️ Stopped")
        self.face_rec.setText("🔴 REC")
        if self.active_module == "face":
            self.active_module = ""
        self.status_bar.showMessage("✅ FaceRec stopped")

    def _face_finished(self) -> None:
        self.face_start.setEnabled(True)
        self.face_stop.setEnabled(False)
        if self.face_stat_lbl.text() == "⚡ Running…":
            self.face_stat_lbl.setText("✅ Finished")
        if self.active_module == "face":
            self.active_module = ""

    def _face_error(self, msg: str) -> None:
        QMessageBox.critical(self, "FaceRec Error", msg)
        self._face_stop()

    def _face_toggle_rec(self) -> None:
        if not (self.face_engine and self.face_engine.isRunning()):
            return
        stopped = self.face_engine.toggle_recording()
        if stopped:
            self.face_rec.setText("🔴 REC")
            self.status_bar.showMessage(f"💾 Video saved → {OUTPUT_DIR}")
        else:
            self.face_rec.setText("⏹ STOP REC")
            self.status_bar.showMessage("🔴 Recording…")

    def _face_screenshot(self) -> None:
        pix = self.face_video.pixmap()
        if pix and not pix.isNull():
            path = OUTPUT_DIR / f"face_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            if pix.save(str(path)):
                self.status_bar.showMessage(f"📸 Saved: {path.name}")

    def _face_save_json(self) -> None:
        if not self.face_history:
            QMessageBox.warning(self, "No Data", "No face data to export yet")
            return
        self._save_json(self.face_history, "face_report")

    def _face_frame(self, img) -> None:
        self.face_video.setPixmap(img_to_pixmap(img, self.face_video.size()))

    def _face_stats(self, fps: float, lat: float, cnt: int) -> None:
        self.face_stat_lbl.setText(f"FPS: {fps:.1f} | Lat: {lat:.1f}ms | Faces: {cnt}")

    def _face_data(self, data: list) -> None:
        self._extend_capped(self.face_history, data)
        if not self._throttle_ui():
            return
        if data:
            lines = [
                f"• {d.get('name','?')} (sim {d.get('similarity',0):.2f})"
                for d in data[:10]
            ]
            self.face_results.setText("\n".join(lines))

    # ══════════════════════════════════════════════════════════ Gesture logic
    def _gesture_src_changed(self, idx: int) -> None:
        self.gesture_file_btn.setVisible(idx > 0)
        if idx == 0:
            self.gesture_config["source_path"] = ""
            self.gesture_stat_lbl.setText("📹 Webcam selected")

    def _gesture_pick_file(self) -> None:
        idx = self.gesture_src.currentIndex()
        ext = ["", "Videos (*.mp4 *.avi *.mkv *.mov)", "Images (*.jpg *.jpeg *.png *.bmp)"][idx]
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", ext)
        if path:
            self.gesture_config["source_path"] = path
            self.gesture_stat_lbl.setText(f"📄 {pathlib.Path(path).name}")

    def _gesture_start(self) -> None:
        if self.active_module == "gesture":
            self._gesture_stop()
        try:
            src_idx = self.gesture_src.currentIndex()
            self.gesture_config.update({
                "source_type":               ["webcam", "video", "image"][src_idx],
                "min_detection_confidence":  self.gesture_sld_det.value()   / 100.0,
                "min_tracking_confidence":   self.gesture_sld_track.value() / 100.0,
                "max_num_hands":             int(self.gesture_hands.text() or "2"),
            })
            self.gesture_history.clear()
            self.gesture_results.clear()
            self.gesture_engine = GestureRecognitionEngine(**self.gesture_config)
            self.gesture_engine.frame_ready.connect(self._gesture_frame)
            self.gesture_engine.stats_ready.connect(self._gesture_stats)
            self.gesture_engine.data_ready.connect(self._gesture_data)
            self.gesture_engine.error_occurred.connect(self._gesture_error)
            self.gesture_engine.finished.connect(self._gesture_finished)
            self.gesture_engine.start()
            self.active_module = "gesture"
            self.gesture_start.setEnabled(False)
            self.gesture_stop.setEnabled(True)
            self.gesture_stat_lbl.setText("⚡ Running…")
            self.status_bar.showMessage("🔥 Gesture Recognition running")
        except Exception as exc:
            logger.error(f"❌ Gesture start: {exc}")
            QMessageBox.critical(self, "Error", str(exc))
            self._gesture_stop()

    def _gesture_stop(self) -> None:
        if self.gesture_engine and self.gesture_engine.isRunning():
            self.gesture_engine.stop()
        self.gesture_start.setEnabled(True)
        self.gesture_stop.setEnabled(False)
        self.gesture_stat_lbl.setText("⏹️ Stopped")
        self.gesture_rec.setText("🔴 REC")
        if self.active_module == "gesture":
            self.active_module = ""
        self.status_bar.showMessage("✅ Gesture recognition stopped")

    def _gesture_finished(self) -> None:
        self.gesture_start.setEnabled(True)
        self.gesture_stop.setEnabled(False)
        if self.gesture_stat_lbl.text() == "⚡ Running…":
            self.gesture_stat_lbl.setText("✅ Finished")
        if self.active_module == "gesture":
            self.active_module = ""

    def _gesture_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Gesture Error", msg)
        self._gesture_stop()

    def _gesture_toggle_rec(self) -> None:
        if not (self.gesture_engine and self.gesture_engine.isRunning()):
            return
        now_recording = self.gesture_engine.toggle_recording()
        if now_recording:
            self.gesture_rec.setText("⏹ STOP REC")
            self.status_bar.showMessage("🔴 Recording…")
        else:
            self.gesture_rec.setText("🔴 REC")
            self.status_bar.showMessage(f"💾 Video saved → {OUTPUT_DIR}")

    def _gesture_screenshot(self) -> None:
        pix = self.gesture_video.pixmap()
        if pix and not pix.isNull():
            path = OUTPUT_DIR / f"gesture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            if pix.save(str(path)):
                self.status_bar.showMessage(f"📸 Saved: {path.name}")

    def _gesture_save_json(self) -> None:
        if not self.gesture_history:
            QMessageBox.warning(self, "No Data", "No gesture data to export yet")
            return
        self._save_json(self.gesture_history, "gesture_report")

    def _gesture_frame(self, img) -> None:
        self.gesture_video.setPixmap(
            img_to_pixmap(img, self.gesture_video.size())
        )

    def _gesture_stats(self, fps: float, lat: float, cnt: int) -> None:
        self.gesture_stat_lbl.setText(f"FPS: {fps:.1f} | Lat: {lat:.1f}ms | Hands: {cnt}")

    def _gesture_data(self, data: list) -> None:
        self._extend_capped(self.gesture_history, data)
        if not self._throttle_ui():
            return
        if data:
            lines = [
                f"• {d.get('handedness','?')} — {d.get('gesture','?')} "
                f"({d.get('confidence',0):.0%})"
                for d in data[:10]
            ]
            self.gesture_results.setText("\n".join(lines))

    # ══════════════════════════════════════════════════════════ Shared helpers
    def _throttle_ui(self, interval: float = 0.25) -> bool:
        """Return True (and update timestamp) if enough time has passed."""
        now = time.monotonic()
        if now - self._last_ui_update < interval:
            return False
        self._last_ui_update = now
        return True

    @staticmethod
    def _extend_capped(lst: list, new_items: list) -> None:
        """Append new items to *lst*, trimming to _MAX_HISTORY total."""
        lst.extend(new_items)
        if len(lst) > _MAX_HISTORY:
            del lst[:len(lst) - _MAX_HISTORY]

    def _save_json(self, data: list, prefix: str) -> None:
        try:
            path = OUTPUT_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
            QMessageBox.information(self, "Saved", f"Report saved:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Save failed: {exc}")

    # ══════════════════════════════════════════════════════════ Close event
    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_all()
        logger.info("👋 Application closed")
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
def img_to_pixmap(q_img, target_size) -> QPixmap:
    """Scale a QImage to *target_size* preserving aspect ratio."""
    return QPixmap.fromImage(q_img).scaled(
        target_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        * { font-family: 'Segoe UI', sans-serif; }
        QMainWindow, QWidget { background: #1a1a1a; color: #eee; }
        QLabel  { color: #ddd; }
        QGroupBox { border: 1px solid #333; border-radius: 6px; margin-top: 8px; }
        QComboBox, QLineEdit {
            background: #2a2a2a; border: 1px solid #444;
            border-radius: 4px; color: #fff; padding: 4px;
        }
        QPushButton { border: none; border-radius: 5px; padding: 6px 12px; }
        QTextEdit   {
            background: #111; border: 1px solid #333;
            border-radius: 4px; font-family: Consolas; font-size: 11px;
        }
        QScrollBar { background: #222; width: 10px; }
        QScrollBar::handle { background: #555; border-radius: 4px; }
        QTabWidget::pane { border: 1px solid #444; background: #1a1a1a; }
        QTabBar::tab {
            background: #2a2a2a; color: #aaa;
            padding: 8px 16px;
            border-top-left-radius: 6px; border-top-right-radius: 6px;
        }
        QTabBar::tab:selected { background: #3a3a3a; color: #fff; font-weight: 600; }
        QSlider::groove:horizontal {
            background: #2a2a2a; border: 1px solid #444;
            height: 6px; border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #3498db; border: none;
            width: 14px; margin: -4px 0; border-radius: 7px;
        }
    """)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
