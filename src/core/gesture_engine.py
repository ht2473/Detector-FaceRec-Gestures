"""MediaPipe hand-gesture recognition engine — runs in a dedicated QThread."""
from __future__ import annotations

import pathlib
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None  # type: ignore[assignment]

BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"

# ── Gesture look-up table (finger states → name) ─────────────────────────────
# Finger order: [Thumb, Index, Middle, Ring, Pinky]
_GESTURE_MAP: Dict[Tuple[int, ...], str] = {
    (0, 0, 0, 0, 0): "Fist",
    (1, 1, 1, 1, 1): "Open Palm",
    (0, 1, 0, 0, 0): "Pointing",
    (0, 1, 1, 0, 0): "Victory / Peace",
    (1, 0, 0, 0, 0): "Thumbs Up",
    (1, 1, 0, 0, 1): "I Love You",
    (0, 1, 0, 0, 1): "Rock On",
    (1, 1, 1, 1, 0): "Four",
    (0, 1, 1, 1, 1): "Four (no thumb)",
    (1, 0, 0, 0, 1): "Call Me",
    (0, 0, 0, 0, 1): "Pinky",
}


def _frame_to_qimage(frame_bgr: np.ndarray) -> QImage:
    if frame_bgr is None or frame_bgr.size == 0:
        return QImage()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


class GestureRecognitionEngine(QThread):
    """Detects hand landmarks and classifies gestures using MediaPipe Hands."""

    frame_ready    = pyqtSignal(QImage)
    stats_ready    = pyqtSignal(float, float, int)  # fps, latency_ms, hand_count
    data_ready     = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        source_type:               str   = "webcam",
        source_path:               str   = "",
        device:                    str   = "cpu",
        min_detection_confidence:  float = 0.5,
        min_tracking_confidence:   float = 0.5,
        max_num_hands:             int   = 2,
        **_kwargs,                          # absorb unused keys from config dict
    ) -> None:
        super().__init__()
        self.source_type = source_type.strip().lower()
        self.source_path = source_path.strip()
        self.min_det_conf   = min_detection_confidence
        self.min_track_conf = min_tracking_confidence
        self.max_hands      = max(1, max_num_hands)

        self.running      = False
        self._detector    = None
        self._cap:        Optional[cv2.VideoCapture] = None
        self._writer:     Optional[cv2.VideoWriter]  = None
        self._recording   = False
        self._record_lock = threading.Lock()

        if MEDIAPIPE_AVAILABLE:
            self._mp_hands      = mp.solutions.hands
            self._mp_draw       = mp.solutions.drawing_utils
            self._mp_draw_styles = mp.solutions.drawing_styles

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        if not MEDIAPIPE_AVAILABLE:
            self.error_occurred.emit(
                "mediapipe is not installed.  Run: pip install mediapipe"
            )
            return

        try:
            self._detector = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                model_complexity=1,
                min_detection_confidence=self.min_det_conf,
                min_tracking_confidence=self.min_track_conf,
            )
            logger.info(
                f"🖐️  MediaPipe Hands ready | max_hands={self.max_hands} | "
                f"source={self.source_type}"
            )
        except Exception as exc:
            self.error_occurred.emit(f"MediaPipe init error: {exc}")
            return

        self.running = True
        try:
            if self.source_type == "image":
                self._process_image()
            else:
                self._stream_loop()
        except Exception as exc:
            logger.exception("Runtime error in gesture loop")
            self.error_occurred.emit(f"Runtime error: {exc}")
        finally:
            self._cleanup()

    # ------------------------------------------------------------------ source handlers
    def _process_image(self) -> None:
        frame = cv2.imread(self.source_path)
        if frame is None:
            self.error_occurred.emit(f"Cannot load image: {self.source_path}")
            return
        _, count, data, out = self._run_inference(frame)
        self.stats_ready.emit(0.0, 0.0, count)
        self.data_ready.emit(data)
        self.frame_ready.emit(_frame_to_qimage(out))

    def _stream_loop(self) -> None:
        is_video = self.source_type in ("video", "video file")

        src = self.source_path if is_video else 0
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            self.error_occurred.emit("Failed to open video source")
            return

        if not is_video:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        raw_fps    = self._cap.get(cv2.CAP_PROP_FPS)
        source_fps = raw_fps if raw_fps and raw_fps > 0 else 30.0
        frame_interval = 1.0 / source_fps

        frame_count = 0
        start_time  = time.time()

        while self.running:
            t_loop = time.perf_counter()

            ret, frame = self._cap.read()
            if not ret or frame is None:
                if is_video:
                    break
                time.sleep(0.005)
                continue

            frame_count += 1
            latency, count, data, out = self._run_inference(frame)

            with self._record_lock:
                if self._recording:
                    if self._writer is None:
                        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                        h, w = out.shape[:2]
                        path = OUTPUT_DIR / f"gesture_{int(time.time())}.mp4"
                        self._writer = cv2.VideoWriter(
                            str(path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            source_fps,
                            (w, h),
                        )
                    self._writer.write(out)

            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            self.frame_ready.emit(_frame_to_qimage(out))
            self.stats_ready.emit(fps, latency, count)
            self.data_ready.emit(data)

            # pace video playback
            if is_video:
                sleep = max(0.0, frame_interval - (time.perf_counter() - t_loop))
                if sleep > 0:
                    time.sleep(sleep)

    # ------------------------------------------------------------------ inference
    @staticmethod
    def _fingers_up(landmarks, raw_label: str) -> Tuple[int, ...]:
        """Return (Thumb, Index, Middle, Ring, Pinky) finger states (0/1)."""
        # tips/PIPs for 4 non-thumb fingers
        tip_ids = (8, 12, 16, 20)
        pip_ids = (6, 10, 14, 18)

        fingers = [0] * 5

        # 4 fingers: extended if tip.y < pip.y (screen coords)
        for i, (tip, pip) in enumerate(zip(tip_ids, pip_ids)):
            fingers[i + 1] = 1 if landmarks[tip].y < landmarks[pip].y else 0

        # Thumb: compare x with MCP (landmark 3)
        # MediaPipe labels from wearer's POV, so "Right" hand appears on left of mirror
        if raw_label == "Right":
            fingers[0] = 1 if landmarks[4].x < landmarks[3].x else 0
        else:
            fingers[0] = 1 if landmarks[4].x > landmarks[3].x else 0

        return tuple(fingers)

    def _run_inference(
        self, frame: np.ndarray
    ) -> Tuple[float, int, List[Dict], np.ndarray]:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        out     = frame.copy()
        data: List[Dict] = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # draw skeleton
                self._mp_draw.draw_landmarks(
                    out, hand_lm, self._mp_hands.HAND_CONNECTIONS,
                    self._mp_draw_styles.get_default_hand_landmarks_style(),
                    self._mp_draw_styles.get_default_hand_connections_style(),
                )

                raw_label = handedness.classification[0].label
                # MP labels from wearer → invert for mirror view
                hand_type = "Left" if raw_label == "Right" else "Right"
                score     = handedness.classification[0].score

                lm = hand_lm.landmark
                xs  = [p.x for p in lm]
                ys  = [p.y for p in lm]
                x_min = int(min(xs) * w)
                x_max = int(max(xs) * w)
                y_min = int(min(ys) * h)
                y_max = int(max(ys) * h)
                bbox  = [x_min, y_min, x_max - x_min, y_max - y_min]

                fingers = self._fingers_up(lm, raw_label)
                gesture = _GESTURE_MAP.get(fingers, "Unknown")

                label_txt = f"{hand_type}: {gesture}"
                cv2.putText(
                    out, label_txt,
                    (x_min, max(y_min - 10, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
                )

                data.append({
                    "handedness": hand_type,
                    "gesture":    gesture,
                    "fingers":    list(fingers),
                    "confidence": round(float(score), 4),
                    "bbox":       bbox,
                    "timestamp":  datetime.now().isoformat(timespec="milliseconds"),
                })

        latency_ms = (time.perf_counter() - t0) * 1000
        return latency_ms, len(data), data, out

    # ------------------------------------------------------------------ recording
    def toggle_recording(self) -> bool:
        """Toggle recording on/off.  Returns True when recording has just **stopped**."""
        with self._record_lock:
            self._recording = not self._recording
            if not self._recording and self._writer is not None:
                self._writer.release()
                self._writer = None
                logger.success(f"💾 Gesture video saved → {OUTPUT_DIR}")
                return True   # stopped
        return False          # started

    # ------------------------------------------------------------------ cleanup / stop
    def _cleanup(self) -> None:
        self.running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._record_lock:
            if self._writer:
                self._writer.release()
                self._writer = None
        if self._detector:
            self._detector.close()
        logger.info("✅ Gesture engine cleaned up")

    def stop(self) -> None:
        self.running = False
        self.wait(3000)
