"""Gesture Recognition Engine based on MediaPipe - Refactored & Optimized"""
import cv2
import time
import math
import pathlib
import threading
import numpy as np
from typing import Optional, Dict, List, Tuple
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from loguru import logger

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"


class GestureRecognitionEngine(QThread):
    frame_ready = pyqtSignal(QImage)
    stats_ready = pyqtSignal(float, float, int)
    data_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        source_type: str = "webcam",
        source_path: str = "",
        device: str = "cpu",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
        **kwargs
    ):
        super().__init__()
        self.source_type = source_type.strip().lower()
        self.source_path = source_path.strip()
        self.max_hands = max_num_hands
        self.min_det_conf = min_detection_confidence
        self.min_track_conf = min_tracking_confidence

        self.running = False
        self.hands_detector = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.record_lock = threading.Lock()

        # Инициализация утилит отрисовки MediaPipe
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_draw_styles = mp.solutions.drawing_styles

    def _convert_to_qimage(self, frame: np.ndarray) -> QImage:
        if frame is None or frame.size == 0:
            return QImage()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # bytesPerLine = ch * w предотвращает искажения при рендеринге в Qt
        return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

    def run(self) -> None:
        if not MEDIAPIPE_AVAILABLE:
            self.error_occurred.emit("Install dependencies: pip install mediapipe opencv-python PyQt6 loguru")
            return

        try:
            self.hands_detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                model_complexity=1,
                min_detection_confidence=self.min_det_conf,
                min_tracking_confidence=self.min_track_conf
            )
            logger.info(f"Engine MediaPipe started | Mode: {self.source_type}")
        except Exception as e:
            self.error_occurred.emit(f"Init error: {e}")
            return

        self.running = True
        try:
            if self.source_type == "image":
                self._process_image()
            else:
                self._process_stream()
        except Exception as e:
            logger.exception("Runtime error in loop")
            self.error_occurred.emit(f"Runtime error: {e}")
        finally:
            self._cleanup()

    def _process_image(self):
        frame = cv2.imread(self.source_path)
        if frame is None:
            self.error_occurred.emit("Cannot load image")
            return
        _, _, data, out_frame = self._run_inference(frame)
        self.frame_ready.emit(self._convert_to_qimage(out_frame))
        self.data_ready.emit(data)

    def _process_stream(self):
        is_video = self.source_type in ["video", "video file"]
        
        self.cap = cv2.VideoCapture(self.source_path if is_video else 0)
        if not self.cap.isOpened():
            self.error_occurred.emit("Failed to open video source")
            return

        if not is_video:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        fps_source = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps_source or fps_source <= 0 or math.isnan(fps_source):
            fps_source = 30.0
            
        frame_delay = 1.0 / fps_source
        start_time = time.time()
        frame_count = 0

        while self.running:
            loop_start = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            latency, hands_count, data, out_frame = self._run_inference(frame)

            with self.record_lock:
                if self.is_recording:
                    if self.video_writer is None:
                        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                        path = OUTPUT_DIR / f"rec_{int(time.time())}.mp4"
                        h, w = out_frame.shape[:2]
                        self.video_writer = cv2.VideoWriter(
                            str(path),
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps_source,
                            (w, h)
                        )
                    self.video_writer.write(out_frame)

            curr_fps = frame_count / (time.time() - start_time)
            
            self.frame_ready.emit(self._convert_to_qimage(out_frame))
            self.stats_ready.emit(curr_fps, latency, hands_count)
            self.data_ready.emit(data)

            # Искусственная задержка для видеофайлов, чтобы сохранить оригинальную скорость
            if is_video:
                processing_time = time.perf_counter() - loop_start
                sleep_time = frame_delay - processing_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _get_fingers_up(self, landmarks, raw_handedness_label: str) -> List[int]:
        """Определяет состояние пальцев [Thumb, Index, Middle, Ring, Pinky] по ладмаркам MediaPipe"""
        fingers = [0] * 5
        tips_ids = [8, 12, 16, 20]
        pips_ids = [6, 10, 14, 18]

        # Указательный, средний, безымянный, мизинец (сравнение по Y)
        for i, (tip_id, pip_id) in enumerate(zip(tips_ids, pips_ids)):
            if landmarks[tip_id].y < landmarks[pip_id].y:
                fingers[i + 1] = 1

        # Большой палец (сравнение по X с учётом руки)
        # MediaPipe возвращает "Left"/"Right" с точки зрения человека (зеркально для камеры)
        if raw_handedness_label == "Right":
            if landmarks[4].x < landmarks[3].x:
                fingers[0] = 1
        else:  # Left
            if landmarks[4].x > landmarks[3].x:
                fingers[0] = 1

        return fingers

    def _run_inference(self, frame: np.ndarray) -> Tuple[float, int, List[Dict], np.ndarray]:
        t0 = time.perf_counter()
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(rgb_frame)
        out_frame = frame.copy()
        data_list = []
        hands_count = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            hands_count = len(results.multi_hand_landmarks)
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Отрисовка скелета руки
                self.mp_draw.draw_landmarks(
                    out_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style()
                )

                # Данные классификации
                raw_label = handedness.classification[0].label
                # Инвертируем метку для соответствия виду с камеры (Right hand -> Left label в MP)
                hand_type = "Right" if raw_label == "Left" else "Left"
                score = handedness.classification[0].score

                # Вычисление Bounding Box
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                # Определение поднятых пальцев и жеста
                fingers = self._get_fingers_up(hand_landmarks.landmark, raw_label)
                gesture = "Unknown"
                if fingers == [0, 0, 0, 0, 0]:
                    gesture = "Fist"
                elif fingers == [1, 1, 1, 1, 1]:
                    gesture = "Open Palm"
                elif fingers == [0, 1, 0, 0, 0]:
                    gesture = "Pointing Up"
                elif fingers == [0, 1, 1, 0, 0]:
                    gesture = "Victory"
                elif fingers == [1, 0, 0, 0, 0]:
                    gesture = "Thumb Up"
                elif fingers == [1, 1, 0, 0, 1]:
                    gesture = "I Love You"

                # Отрисовка текста
                cv2.putText(out_frame, f"{hand_type}: {gesture}", (bbox[0] + 10, bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                data_list.append({
                    "handedness": hand_type,
                    "gesture": gesture,
                    "confidence": score,
                    "bbox": bbox
                })

        latency = (time.perf_counter() - t0) * 1000
        return latency, hands_count, data_list, out_frame

    def toggle_recording(self) -> bool:
        with self.record_lock:
            self.is_recording = not self.is_recording
            if not self.is_recording and self.video_writer:
                self.video_writer.release()
                self.video_writer = None
        return self.is_recording

    def _cleanup(self):
        self.running = False
        if self.cap:
            self.cap.release()
        with self.record_lock:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
        if self.hands_detector:
            self.hands_detector.close()
        logger.info("✅ Engine resources cleaned up")

    def stop(self):
        self.running = False
        self.wait(2000)