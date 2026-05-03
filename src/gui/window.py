"""Main application window - Detection, Face & Gesture Modules"""
import sys
import time
import pathlib
import json
from datetime import datetime
from typing import Optional, List, Dict
import torch
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QGroupBox, QStatusBar,
    QFileDialog, QTextEdit, QFormLayout, QLineEdit, QMessageBox, QTabWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from src.core.engine import DetectionEngine
from src.core.face_engine import FaceRecognitionEngine
from src.core.gesture_engine import GestureRecognitionEngine
from src.utils.helpers import ensure_dir
from loguru import logger

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detector + FaceRec + Gestures")
        self.setMinimumSize(1280, 850)
        ensure_dir(MODELS_DIR)
        ensure_dir(OUTPUT_DIR)

        self.det_engine: Optional[DetectionEngine] = None
        self.face_engine: Optional[FaceRecognitionEngine] = None
        self.gesture_engine: Optional[GestureRecognitionEngine] = None
        self.source_path: str = ""
        self.face_db_path: str = ""
        self.active_module: str = ""
        self.detection_history: List[Dict] = []
        self.face_history: List[Dict] = []
        self.gesture_history: List[Dict] = []
        self._last_ui_update: float = 0.0

        self.det_config = {
            "source_type": "webcam", "source_path": "", "model_name": "yolo26s.pt",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "conf": 0.25, "iou": 0.45, "imgsz": 640
        }
        self.face_config = {
            "source_type": "webcam", "source_path": "", "device": "cpu",
            "det_thresh": 0.5, "rec_thresh": 0.35, "face_db_dir": "", "imgsz": 320, "skip_frames": 3
        }
        self.gesture_config = {
            "source_type": "webcam", "source_path": "", "device": "cpu",
            "min_detection_confidence": 0.5, "min_tracking_confidence": 0.5,
            "max_num_hands": 2, "imgsz": 640, "skip_frames": 3
        }

        self._init_ui()
        logger.info("🚀 Application started (Detection + FaceRec + Gestures)")

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_detection_tab(), "🔍 Object Detection")
        self.tabs.addTab(self._create_face_tab(), "👤 Face Recognition")
        self.tabs.addTab(self._create_gesture_tab(), "🖐️ Gesture Recognition")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        main_layout.addWidget(self.tabs)
        self.global_status = QStatusBar()
        self.global_status.setStyleSheet("color: #0f0; background: #111; font-weight: 500;")
        self.global_status.showMessage("✅ Ready — Select tab, configure and click START")
        self.setStatusBar(self.global_status)

    def _on_tab_changed(self, idx):
        self._stop_all()
        self.global_status.showMessage("✅ Tab switched — Ready")

    def _stop_all(self):
        if self.det_engine and self.det_engine.isRunning(): self._stop_detection()
        if self.face_engine and self.face_engine.isRunning(): self._stop_face()
        if self.gesture_engine and self.gesture_engine.isRunning(): self._stop_gesture()

    def _group_box(self, title: str) -> QGroupBox:
        gb = QGroupBox(title)
        gb.setStyleSheet("QGroupBox { font-weight: 600; color: #fff; border: 1px solid #444; border-radius: 8px; margin-top: 8px; padding-top: 12px; background: #1e1e1e; } QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }")
        return gb

    def _btn_style(self, bg: str) -> str:
        return f"QPushButton {{ background: {bg}; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: 500; }} QPushButton:hover {{ background: {bg}; opacity: 0.9; border: 1px solid rgba(255,255,255,0.2); }} QPushButton:disabled {{ background: #444; color: #888; }}"

    def _slider_with_label(self, slider: QSlider, label: QLabel) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(slider)
        lay.addWidget(label)
        return w

    # ================= DETECTION TAB =================
    def _create_detection_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.det_video = QLabel("Select source and click START")
        self.det_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.det_video.setStyleSheet("background: #0a0a0a; color: #555; border: 2px solid #333; border-radius: 10px; min-height: 450px; font-size: 14px;")
        left_layout.addWidget(self.det_video)
        self.det_status = QLabel("Idle")
        self.det_status.setStyleSheet("color: #888; font-size: 13px;")
        left_layout.addWidget(self.det_status)
        layout.addWidget(left, 3)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(12)
        grp_src = self._group_box("📁 Source")
        lay_src = QVBoxLayout(grp_src)
        self.det_combo_src = QComboBox()
        self.det_combo_src.addItems(["📹 Webcam", "🎬 Video File", "🖼️ Image"])
        self.det_combo_src.currentIndexChanged.connect(self._on_det_source_change)
        lay_src.addWidget(self.det_combo_src)
        self.det_select_file = QPushButton("📂 Select File...")
        self.det_select_file.hide()
        self.det_select_file.clicked.connect(self._select_det_file)
        lay_src.addWidget(self.det_select_file)
        right_layout.addWidget(grp_src)

        grp_model = self._group_box("🤖 Model & Device")
        lay_mod = QFormLayout(grp_model)
        self.det_combo_model = QComboBox()
        self.det_combo_model.addItems(["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt"])
        self.det_combo_model.setCurrentText("yolo26s.pt")
        lay_mod.addRow("Model:", self.det_combo_model)
        self.det_combo_device = QComboBox()
        self.det_combo_device.addItems(["CPU", "CUDA"])
        if torch.cuda.is_available(): self.det_combo_device.setCurrentIndex(1)
        lay_mod.addRow("Device:", self.det_combo_device)
        right_layout.addWidget(grp_model)

        grp_thr = self._group_box("⚙️ Thresholds")
        lay_thr = QFormLayout(grp_thr)
        self.det_slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.det_slider_conf.setRange(5, 95); self.det_slider_conf.setValue(25)
        self.det_lbl_conf = QLabel("0.25")
        self.det_slider_conf.valueChanged.connect(lambda v: self.det_lbl_conf.setText(f"{v/100:.2f}"))
        lay_thr.addRow("Confidence:", self._slider_with_label(self.det_slider_conf, self.det_lbl_conf))
        self.det_slider_iou = QSlider(Qt.Orientation.Horizontal)
        self.det_slider_iou.setRange(5, 95); self.det_slider_iou.setValue(45)
        self.det_lbl_iou = QLabel("0.45")
        self.det_slider_iou.valueChanged.connect(lambda v: self.det_lbl_iou.setText(f"{v/100:.2f}"))
        lay_thr.addRow("IoU (NMS):", self._slider_with_label(self.det_slider_iou, self.det_lbl_iou))
        self.det_spin_imgsz = QLineEdit("640")
        lay_thr.addRow("Input Size:", self.det_spin_imgsz)
        right_layout.addWidget(grp_thr)

        grp_ctrl = self._group_box("🎮 Control")
        lay_ctrl = QVBoxLayout(grp_ctrl)
        btn_row = QHBoxLayout()
        self.det_start = QPushButton("▶ START")
        self.det_start.setStyleSheet(self._btn_style("#2ecc71"))
        self.det_start.clicked.connect(self._start_detection)
        btn_row.addWidget(self.det_start)
        self.det_stop = QPushButton("⏹ STOP")
        self.det_stop.setEnabled(False)
        self.det_stop.setStyleSheet(self._btn_style("#e74c3c"))
        self.det_stop.clicked.connect(self._stop_detection)
        btn_row.addWidget(self.det_stop)
        lay_ctrl.addLayout(btn_row)
        right_layout.addWidget(grp_ctrl)

        grp_exp = self._group_box("💾 Export")
        lay_exp = QHBoxLayout(grp_exp)
        self.det_rec = QPushButton("🔴 REC")
        self.det_rec.setStyleSheet(self._btn_style("#333"))
        self.det_rec.clicked.connect(self._toggle_det_rec)
        lay_exp.addWidget(self.det_rec)
        self.det_shot = QPushButton("📸 SC")
        self.det_shot.setStyleSheet(self._btn_style("#333"))
        self.det_shot.clicked.connect(self._take_det_screenshot)
        lay_exp.addWidget(self.det_shot)
        self.det_json = QPushButton("📝 JSON")
        self.det_json.setStyleSheet(self._btn_style("#333"))
        self.det_json.clicked.connect(self._save_det_json)
        lay_exp.addWidget(self.det_json)
        right_layout.addWidget(grp_exp)

        grp_res = self._group_box("🎯 Detected Objects")
        lay_res = QVBoxLayout(grp_res)
        self.det_results = QTextEdit()
        self.det_results.setReadOnly(True)
        self.det_results.setStyleSheet("background: #111; color: #0f0; font-family: Consolas; font-size: 11px;")
        lay_res.addWidget(self.det_results)
        right_layout.addWidget(grp_res)
        right_layout.addStretch()
        layout.addWidget(right, 1)
        return tab

    # ================= FACE RECOGNITION TAB =================
    def _create_face_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.face_video = QLabel("Select source and click START")
        self.face_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.face_video.setStyleSheet("background: #0a0a0a; color: #555; border: 2px solid #333; border-radius: 10px; min-height: 450px; font-size: 14px;")
        left_layout.addWidget(self.face_video)
        self.face_status = QLabel("Idle")
        self.face_status.setStyleSheet("color: #888; font-size: 13px;")
        left_layout.addWidget(self.face_status)
        layout.addWidget(left, 3)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(12)
        grp_src = self._group_box("📁 Source")
        lay_src = QVBoxLayout(grp_src)
        self.face_combo_src = QComboBox()
        self.face_combo_src.addItems(["📹 Webcam", "🎬 Video File", "🖼️ Image"])
        self.face_combo_src.currentIndexChanged.connect(self._on_face_source_change)
        lay_src.addWidget(self.face_combo_src)
        self.face_select_file = QPushButton("📂 Select File...")
        self.face_select_file.hide()
        self.face_select_file.clicked.connect(self._select_face_file)
        lay_src.addWidget(self.face_select_file)
        right_layout.addWidget(grp_src)

        grp_db = self._group_box("👥 Face Database")
        lay_db = QVBoxLayout(grp_db)
        self.face_load_db = QPushButton("📂 Load DB Folder...")
        self.face_load_db.clicked.connect(self._load_face_db)
        lay_db.addWidget(self.face_load_db)
        self.face_db_status = QLabel("No DB loaded (All -> Unknown)")
        self.face_db_status.setStyleSheet("color: #aaa; font-size: 12px;")
        lay_db.addWidget(self.face_db_status)
        right_layout.addWidget(grp_db)

        grp_thr = self._group_box("⚙️ Thresholds")
        lay_thr = QFormLayout(grp_thr)
        self.face_slider_det = QSlider(Qt.Orientation.Horizontal)
        self.face_slider_det.setRange(10, 90); self.face_slider_det.setValue(50)
        self.face_lbl_det = QLabel("0.50")
        self.face_slider_det.valueChanged.connect(lambda v: self.face_lbl_det.setText(f"{v/100:.2f}"))
        lay_thr.addRow("Det Conf:", self._slider_with_label(self.face_slider_det, self.face_lbl_det))
        self.face_slider_rec = QSlider(Qt.Orientation.Horizontal)
        self.face_slider_rec.setRange(10, 80); self.face_slider_rec.setValue(35)
        self.face_lbl_rec = QLabel("0.35")
        self.face_slider_rec.valueChanged.connect(lambda v: self.face_lbl_rec.setText(f"{v/100:.2f}"))
        lay_thr.addRow("Rec Sim:", self._slider_with_label(self.face_slider_rec, self.face_lbl_rec))
        right_layout.addWidget(grp_thr)

        grp_ctrl = self._group_box("🎮 Control")
        lay_ctrl = QVBoxLayout(grp_ctrl)
        btn_row = QHBoxLayout()
        self.face_start = QPushButton("▶ START")
        self.face_start.setStyleSheet(self._btn_style("#3498db"))
        self.face_start.clicked.connect(self._start_face)
        btn_row.addWidget(self.face_start)
        self.face_stop = QPushButton("⏹ STOP")
        self.face_stop.setEnabled(False)
        self.face_stop.setStyleSheet(self._btn_style("#e74c3c"))
        self.face_stop.clicked.connect(self._stop_face)
        btn_row.addWidget(self.face_stop)
        lay_ctrl.addLayout(btn_row)
        right_layout.addWidget(grp_ctrl)

        grp_exp = self._group_box("💾 Export")
        lay_exp = QHBoxLayout(grp_exp)
        self.face_rec_btn = QPushButton("🔴 REC")
        self.face_rec_btn.setStyleSheet(self._btn_style("#333"))
        self.face_rec_btn.clicked.connect(self._toggle_face_rec)
        lay_exp.addWidget(self.face_rec_btn)
        self.face_shot = QPushButton("📸 SC")
        self.face_shot.setStyleSheet(self._btn_style("#333"))
        self.face_shot.clicked.connect(self._take_face_screenshot)
        lay_exp.addWidget(self.face_shot)
        self.face_json = QPushButton("📝 JSON")
        self.face_json.setStyleSheet(self._btn_style("#333"))
        self.face_json.clicked.connect(self._save_face_json)
        lay_exp.addWidget(self.face_json)
        right_layout.addWidget(grp_exp)

        grp_res = self._group_box("👤 Recognized Faces")
        lay_res = QVBoxLayout(grp_res)
        self.face_results = QTextEdit()
        self.face_results.setReadOnly(True)
        self.face_results.setStyleSheet("background: #111; color: #0af; font-family: Consolas; font-size: 11px;")
        lay_res.addWidget(self.face_results)
        right_layout.addWidget(grp_res)
        right_layout.addStretch()
        layout.addWidget(right, 1)
        return tab

    # ================= GESTURE RECOGNITION TAB =================
    def _create_gesture_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.gesture_video = QLabel("Select source and click START")
        self.gesture_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gesture_video.setStyleSheet("background: #0a0a0a; color: #555; border: 2px solid #333; border-radius: 10px; min-height: 450px; font-size: 14px;")
        left_layout.addWidget(self.gesture_video)
        self.gesture_status = QLabel("Idle")
        self.gesture_status.setStyleSheet("color: #888; font-size: 13px;")
        left_layout.addWidget(self.gesture_status)
        layout.addWidget(left, 3)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(12)
        grp_src = self._group_box("📁 Source")
        lay_src = QVBoxLayout(grp_src)
        self.gesture_combo_src = QComboBox()
        self.gesture_combo_src.addItems(["📹 Webcam", "🎬 Video File", "🖼️ Image"])
        self.gesture_combo_src.currentIndexChanged.connect(self._on_gesture_source_change)
        lay_src.addWidget(self.gesture_combo_src)
        self.gesture_select_file = QPushButton("📂 Select File...")
        self.gesture_select_file.hide()
        self.gesture_select_file.clicked.connect(self._select_gesture_file)
        lay_src.addWidget(self.gesture_select_file)
        right_layout.addWidget(grp_src)

        grp_thr = self._group_box("⚙️ MediaPipe Settings")
        lay_thr = QFormLayout(grp_thr)
        self.gesture_slider_det = QSlider(Qt.Orientation.Horizontal)
        self.gesture_slider_det.setRange(10, 90); self.gesture_slider_det.setValue(50)
        self.gesture_lbl_det = QLabel("0.50")
        self.gesture_slider_det.valueChanged.connect(lambda v: self.gesture_lbl_det.setText(f"{v/100:.2f}"))
        lay_thr.addRow("Det Conf:", self._slider_with_label(self.gesture_slider_det, self.gesture_lbl_det))
        self.gesture_slider_track = QSlider(Qt.Orientation.Horizontal)
        self.gesture_slider_track.setRange(10, 90); self.gesture_slider_track.setValue(50)
        self.gesture_lbl_track = QLabel("0.50")
        self.gesture_slider_track.valueChanged.connect(lambda v: self.gesture_lbl_track.setText(f"{v/100:.2f}"))
        lay_thr.addRow("Track Conf:", self._slider_with_label(self.gesture_slider_track, self.gesture_lbl_track))
        self.gesture_spin_hands = QLineEdit("2")
        lay_thr.addRow("Max Hands:", self.gesture_spin_hands)
        right_layout.addWidget(grp_thr)

        grp_ctrl = self._group_box("🎮 Control")
        lay_ctrl = QVBoxLayout(grp_ctrl)
        btn_row = QHBoxLayout()
        self.gesture_start = QPushButton("▶ START")
        self.gesture_start.setStyleSheet(self._btn_style("#9b59b6"))
        self.gesture_start.clicked.connect(self._start_gesture)
        btn_row.addWidget(self.gesture_start)
        self.gesture_stop = QPushButton("⏹ STOP")
        self.gesture_stop.setEnabled(False)
        self.gesture_stop.setStyleSheet(self._btn_style("#e74c3c"))
        self.gesture_stop.clicked.connect(self._stop_gesture)
        btn_row.addWidget(self.gesture_stop)
        lay_ctrl.addLayout(btn_row)
        right_layout.addWidget(grp_ctrl)

        grp_exp = self._group_box("💾 Export")
        lay_exp = QHBoxLayout(grp_exp)
        self.gesture_rec_btn = QPushButton("🔴 REC")
        self.gesture_rec_btn.setStyleSheet(self._btn_style("#333"))
        self.gesture_rec_btn.clicked.connect(self._toggle_gesture_rec)
        lay_exp.addWidget(self.gesture_rec_btn)
        self.gesture_shot = QPushButton("📸 SC")
        self.gesture_shot.setStyleSheet(self._btn_style("#333"))
        self.gesture_shot.clicked.connect(self._take_gesture_screenshot)
        lay_exp.addWidget(self.gesture_shot)
        self.gesture_json = QPushButton("📝 JSON")
        self.gesture_json.setStyleSheet(self._btn_style("#333"))
        self.gesture_json.clicked.connect(self._save_gesture_json)
        lay_exp.addWidget(self.gesture_json)
        right_layout.addWidget(grp_exp)

        grp_res = self._group_box("🖐️ Recognized Gestures")
        lay_res = QVBoxLayout(grp_res)
        self.gesture_results = QTextEdit()
        self.gesture_results.setReadOnly(True)
        self.gesture_results.setStyleSheet("background: #111; color: #d4a5ff; font-family: Consolas; font-size: 11px;")
        lay_res.addWidget(self.gesture_results)
        right_layout.addWidget(grp_res)
        right_layout.addStretch()
        layout.addWidget(right, 1)
        return tab

    # ================= DETECTION LOGIC =================
    def _on_det_source_change(self, idx):
        self.det_select_file.setVisible(idx > 0)
        if idx == 0:
            self.source_path = ""
            self.det_status.setText("📹 Webcam selected")

    def _select_det_file(self):
        idx = self.det_combo_src.currentIndex()
        if idx == 0: return
        title = ["", "Video", "Image"][idx]
        ext = ["", "Videos (*.mp4 *.avi *.mkv)", "Images (*.jpg *.png)"][idx]
        file, _ = QFileDialog.getOpenFileName(self, f"Select {title}", "", ext)
        if file:
            self.source_path = file
            self.det_status.setText(f"📄 {pathlib.Path(file).name}")

    def _start_detection(self):
        if self.active_module == "det": self._stop_detection()
        try:
            source_type = ["webcam", "video", "image"][self.det_combo_src.currentIndex()]
            self.det_config.update({
                "source_type": source_type,
                "source_path": self.source_path if source_type in ("video", "image") else "",
                "model_name": self.det_combo_model.currentText(),
                "device": self.det_combo_device.currentText().lower(),
                "conf": self.det_slider_conf.value() / 100.0,
                "iou": self.det_slider_iou.value() / 100.0,
                "imgsz": int(self.det_spin_imgsz.text() or 640)
            })
            self.det_engine = DetectionEngine(**self.det_config)
            self.detection_history.clear()
            self.det_results.clear()
            self.det_engine.frame_ready.connect(self._update_det_frame)
            self.det_engine.stats_ready.connect(self._update_det_stats)
            self.det_engine.data_ready.connect(self._update_det_results)
            self.det_engine.error_occurred.connect(self._handle_det_error)
            self.det_engine.finished.connect(self._on_det_finished)
            self.det_engine.start()
            self.active_module = "det"
            self.det_start.setEnabled(False)
            self.det_stop.setEnabled(True)
            self.det_status.setText("⚡ Processing...")
            self.global_status.showMessage("🔥 Detection module running")
        except Exception as e:
            logger.error(f"❌ Det start failed: {e}")
            QMessageBox.critical(self, "Error", str(e))
            self._stop_detection()

    def _stop_detection(self):
        if self.det_engine and self.det_engine.isRunning(): self.det_engine.stop()
        self.det_start.setEnabled(True)
        self.det_stop.setEnabled(False)
        self.det_status.setText("⏹️ Stopped")
        if self.det_rec.text() == "⏹ STOP":
            self.det_rec.setText("🔴 REC")
            self.det_rec.setStyleSheet(self._btn_style("#333"))
        if self.active_module == "det": self.active_module = ""
        self.global_status.showMessage("✅ Detection stopped — Ready")

    def _on_det_finished(self):
        self.det_start.setEnabled(True)
        self.det_stop.setEnabled(False)
        if self.det_status.text() == "⚡ Processing...": self.det_status.setText("✅ Finished")
        if self.active_module == "det": self.active_module = ""

    def _handle_det_error(self, msg):
        QMessageBox.critical(self, "Detection Error", msg)
        self._stop_detection()

    def _toggle_det_rec(self):
        if self.det_engine and self.det_engine.isRunning():
            if self.det_engine.toggle_recording():
                self.det_rec.setText("🔴 REC")
                self.det_rec.setStyleSheet(self._btn_style("#333"))
                QMessageBox.information(self, "Ready", f"Video saved to {OUTPUT_DIR}")
            else:
                self.det_rec.setText("⏹ STOP")
                self.det_rec.setStyleSheet(self._btn_style("#e74c3c"))

    def _take_det_screenshot(self):
        pix = self.det_video.pixmap()
        if pix and not pix.isNull():
            path = OUTPUT_DIR / f"det_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            if pix.save(str(path)): self.global_status.showMessage(f"📸 Saved: {path.name}")
            else: QMessageBox.critical(self, "Error", "Failed to save screenshot")

    def _save_det_json(self):
        if not self.detection_history:
            QMessageBox.warning(self, "No Data", "No detections to export")
            return
        try:
            path = OUTPUT_DIR / f"detection_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.detection_history, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Success", f"Report saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")

    def _update_det_frame(self, q_img):
        self.det_video.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.det_video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))

    def _update_det_stats(self, fps, lat, cnt):
        self.det_status.setText(f"FPS: {fps:.1f} | Lat: {lat:.1f}ms | Obj: {cnt}")

    def _update_det_results(self, data):
        self.detection_history.extend(data[:1000])
        now = time.time()
        if now - self._last_ui_update < 0.25: return
        self._last_ui_update = now
        if not data: return
        lines = [f"• {d.get('class_name','?')} ({d.get('confidence',0):.0%}) [{d.get('bbox',[])}]" for d in data[:10]]
        self.det_results.setText("\n".join(lines))

    # ================= FACE RECOGNITION LOGIC =================
    def _on_face_source_change(self, idx):
        self.face_select_file.setVisible(idx > 0)
        if idx == 0:
            self.face_config["source_path"] = ""
            self.face_status.setText("📹 Webcam selected")

    def _select_face_file(self):
        idx = self.face_combo_src.currentIndex()
        if idx == 0: return
        title = ["", "Video", "Image"][idx]
        ext = ["", "Videos (*.mp4 *.avi *.mkv)", "Images (*.jpg *.png)"][idx]
        file, _ = QFileDialog.getOpenFileName(self, f"Select {title}", "", ext)
        if file:
            self.face_config["source_path"] = file
            self.face_status.setText(f"📄 {pathlib.Path(file).name}")

    def _load_face_db(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Face DB Folder")
        if folder:
            self.face_db_path = folder
            self.face_config["face_db_dir"] = folder
            self.face_db_status.setText(f"✅ DB: {pathlib.Path(folder).name} ({len(list(pathlib.Path(folder).glob('*.[jp][pn]g')))} imgs)")

    def _start_face(self):
        if self.active_module == "face": self._stop_face()
        try:
            source_type = ["webcam", "video", "image"][self.face_combo_src.currentIndex()]
            self.face_config.update({
                "source_type": source_type,
                "source_path": self.face_config.get("source_path", ""),
                "device": "cpu",
                "det_thresh": self.face_slider_det.value() / 100.0,
                "rec_thresh": self.face_slider_rec.value() / 100.0,
                "face_db_dir": self.face_config.get("face_db_dir", ""),
                "imgsz": 320,
                "skip_frames": 3
            })
            self.face_engine = FaceRecognitionEngine(**self.face_config)
            self.face_history.clear()
            self.face_results.clear()
            self.face_engine.frame_ready.connect(self._update_face_frame)
            self.face_engine.stats_ready.connect(self._update_face_stats)
            self.face_engine.data_ready.connect(self._update_face_results)
            self.face_engine.error_occurred.connect(self._handle_face_error)
            self.face_engine.finished.connect(self._on_face_finished)
            self.face_engine.start()
            self.active_module = "face"
            self.face_start.setEnabled(False)
            self.face_stop.setEnabled(True)
            self.face_status.setText("⚡ Processing...")
            self.global_status.showMessage("🔥 Face Recognition running")
        except Exception as e:
            logger.error(f"❌ Face start failed: {e}")
            QMessageBox.critical(self, "Error", str(e))
            self._stop_face()

    def _stop_face(self):
        if self.face_engine and self.face_engine.isRunning(): self.face_engine.stop()
        self.face_start.setEnabled(True)
        self.face_stop.setEnabled(False)
        self.face_status.setText("⏹️ Stopped")
        if self.face_rec_btn.text() == "⏹ STOP":
            self.face_rec_btn.setText("🔴 REC")
            self.face_rec_btn.setStyleSheet(self._btn_style("#333"))
        if self.active_module == "face": self.active_module = ""
        self.global_status.showMessage("✅ FaceRec stopped — Ready")

    def _on_face_finished(self):
        self.face_start.setEnabled(True)
        self.face_stop.setEnabled(False)
        if self.face_status.text() == "⚡ Processing...": self.face_status.setText("✅ Finished")
        if self.active_module == "face": self.active_module = ""

    def _handle_face_error(self, msg):
        QMessageBox.critical(self, "FaceRec Error", msg)
        self._stop_face()

    def _toggle_face_rec(self):
        if self.face_engine and self.face_engine.isRunning():
            if self.face_engine.toggle_recording():
                self.face_rec_btn.setText("🔴 REC")
                self.face_rec_btn.setStyleSheet(self._btn_style("#333"))
                QMessageBox.information(self, "Ready", f"Video saved to {OUTPUT_DIR}")
            else:
                self.face_rec_btn.setText("⏹ STOP")
                self.face_rec_btn.setStyleSheet(self._btn_style("#e74c3c"))

    def _take_face_screenshot(self):
        pix = self.face_video.pixmap()
        if pix and not pix.isNull():
            path = OUTPUT_DIR / f"face_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            if pix.save(str(path)): self.global_status.showMessage(f"📸 Saved: {path.name}")
            else: QMessageBox.critical(self, "Error", "Failed to save screenshot")

    def _save_face_json(self):
        if not self.face_history:
            QMessageBox.warning(self, "No Data", "No face data to export")
            return
        try:
            path = OUTPUT_DIR / f"face_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.face_history, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Success", f"Report saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")

    def _update_face_frame(self, q_img):
        self.face_video.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.face_video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))

    def _update_face_stats(self, fps, lat, cnt):
        self.face_status.setText(f"FPS: {fps:.1f} | Lat: {lat:.1f}ms | Faces: {cnt}")

    def _update_face_results(self, data):
        self.face_history.extend(data[:1000])
        now = time.time()
        if now - self._last_ui_update < 0.25: return
        self._last_ui_update = now
        if not data: return
        lines = [f"• {d.get('name','?')} (Sim: {d.get('similarity',0):.2f}) [{d.get('bbox',[])}]" for d in data[:10]]
        self.face_results.setText("\n".join(lines))

    # ================= GESTURE RECOGNITION LOGIC =================
    def _on_gesture_source_change(self, idx):
        self.gesture_select_file.setVisible(idx > 0)
        if idx == 0:
            self.gesture_config["source_path"] = ""
            self.gesture_status.setText("📹 Webcam selected")

    def _select_gesture_file(self):
        idx = self.gesture_combo_src.currentIndex()
        if idx == 0: return
        title = ["", "Video", "Image"][idx]
        ext = ["", "Videos (*.mp4 *.avi *.mkv)", "Images (*.jpg *.png)"][idx]
        file, _ = QFileDialog.getOpenFileName(self, f"Select {title}", "", ext)
        if file:
            self.gesture_config["source_path"] = file
            self.gesture_status.setText(f"📄 {pathlib.Path(file).name}")

    def _start_gesture(self):
        if self.active_module == "gesture": self._stop_gesture()
        try:
            source_type = ["webcam", "video", "image"][self.gesture_combo_src.currentIndex()]
            self.gesture_config.update({
                "source_type": source_type,
                "source_path": self.gesture_config.get("source_path", ""),
                "device": "cpu",
                "min_detection_confidence": self.gesture_slider_det.value() / 100.0,
                "min_tracking_confidence": self.gesture_slider_track.value() / 100.0,
                "max_num_hands": int(self.gesture_spin_hands.text() or 2),
                "imgsz": 640,
                "skip_frames": 3
            })
            self.gesture_engine = GestureRecognitionEngine(**self.gesture_config)
            self.gesture_history.clear()
            self.gesture_results.clear()
            self.gesture_engine.frame_ready.connect(self._update_gesture_frame)
            self.gesture_engine.stats_ready.connect(self._update_gesture_stats)
            self.gesture_engine.data_ready.connect(self._update_gesture_results)
            self.gesture_engine.error_occurred.connect(self._handle_gesture_error)
            self.gesture_engine.finished.connect(self._on_gesture_finished)
            self.gesture_engine.start()
            self.active_module = "gesture"
            self.gesture_start.setEnabled(False)
            self.gesture_stop.setEnabled(True)
            self.gesture_status.setText("⚡ Processing...")
            self.global_status.showMessage("🔥 Gesture Recognition running")
        except Exception as e:
            logger.error(f"❌ Gesture start failed: {e}")
            QMessageBox.critical(self, "Error", str(e))
            self._stop_gesture()

    def _stop_gesture(self):
        if self.gesture_engine and self.gesture_engine.isRunning(): self.gesture_engine.stop()
        self.gesture_start.setEnabled(True)
        self.gesture_stop.setEnabled(False)
        self.gesture_status.setText("⏹️ Stopped")
        if self.gesture_rec_btn.text() == "⏹ STOP":
            self.gesture_rec_btn.setText("🔴 REC")
            self.gesture_rec_btn.setStyleSheet(self._btn_style("#333"))
        if self.active_module == "gesture": self.active_module = ""
        self.global_status.showMessage("✅ GestureRec stopped — Ready")

    def _on_gesture_finished(self):
        self.gesture_start.setEnabled(True)
        self.gesture_stop.setEnabled(False)
        if self.gesture_status.text() == "⚡ Processing...": self.gesture_status.setText("✅ Finished")
        if self.active_module == "gesture": self.active_module = ""

    def _handle_gesture_error(self, msg):
        QMessageBox.critical(self, "GestureRec Error", msg)
        self._stop_gesture()

    def _toggle_gesture_rec(self):
        if self.gesture_engine and self.gesture_engine.isRunning():
            if self.gesture_engine.toggle_recording():
                self.gesture_rec_btn.setText("🔴 REC")
                self.gesture_rec_btn.setStyleSheet(self._btn_style("#333"))
                QMessageBox.information(self, "Ready", f"Video saved to {OUTPUT_DIR}")
            else:
                self.gesture_rec_btn.setText("⏹ STOP")
                self.gesture_rec_btn.setStyleSheet(self._btn_style("#e74c3c"))

    def _take_gesture_screenshot(self):
        pix = self.gesture_video.pixmap()
        if pix and not pix.isNull():
            path = OUTPUT_DIR / f"gesture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            if pix.save(str(path)): self.global_status.showMessage(f"📸 Saved: {path.name}")
            else: QMessageBox.critical(self, "Error", "Failed to save screenshot")

    def _save_gesture_json(self):
        if not self.gesture_history:
            QMessageBox.warning(self, "No Data", "No gesture data to export")
            return
        try:
            path = OUTPUT_DIR / f"gesture_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.gesture_history, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Success", f"Report saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")

    def _update_gesture_frame(self, q_img):
        self.gesture_video.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.gesture_video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))

    def _update_gesture_stats(self, fps, lat, cnt):
        self.gesture_status.setText(f"FPS: {fps:.1f} | Lat: {lat:.1f}ms | Hands: {cnt}")

    def _update_gesture_results(self, data):
        self.gesture_history.extend(data[:1000])
        now = time.time()
        if now - self._last_ui_update < 0.25: return
        self._last_ui_update = now
        if not data: return
        lines = [f"• {d.get('handedness','?')} Hand: {d.get('gesture','?')} ({d.get('confidence',0):.0%})" for d in data[:10]]
        self.gesture_results.setText("\n".join(lines))

    def closeEvent(self, event):
        self._stop_all()
        logger.info("👋 Application closed")
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        * { font-family: 'Segoe UI', sans-serif; }
        QMainWindow, QWidget { background: #1a1a1a; color: #eee; }
        QLabel { color: #ddd; }
        QGroupBox { border: 1px solid #333; border-radius: 6px; margin-top: 8px; }
        QComboBox, QLineEdit, QSlider::groove { background: #2a2a2a; border: 1px solid #444; border-radius: 4px; color: #fff; padding: 4px; }
        QPushButton { border: none; border-radius: 5px; padding: 6px 12px; }
        QTextEdit { background: #111; border: 1px solid #333; border-radius: 4px; font-family: Consolas; font-size: 11px; }
        QScrollBar { background: #222; width: 10px; }
        QScrollBar::handle { background: #555; border-radius: 4px; }
        QTabWidget::pane { border: 1px solid #444; background: #1a1a1a; }
        QTabBar::tab { background: #2a2a2a; color: #aaa; padding: 8px 16px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
        QTabBar::tab:selected { background: #3a3a3a; color: #fff; font-weight: 600; }
    """)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()