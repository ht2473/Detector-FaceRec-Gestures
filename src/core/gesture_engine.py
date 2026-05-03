"""Gesture Recognition Engine based on MediaPipe Hands (Stable Version)"""
import cv2
import time
import pathlib
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from loguru import logger

MEDIAPIPE_AVAILABLE = False
mp_hands = None
mp_drawing = None
mp_drawing_styles = None

try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        MEDIAPIPE_AVAILABLE = True
        logger.success("✅ MediaPipe initialized successfully")
    else:
        raise AttributeError("MediaPipe module is incomplete")
except Exception as e:
    logger.error(f"❌ Failed to initialize MediaPipe: {e}")

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        imgsz: int = 640,
        skip_frames: int = 1  # Установлено в 1 для устранения мерцания
    ):
        super().__init__()
        self.source_type = source_type.strip()
        self.source_path = source_path.strip()
        self.min_det_conf = min_detection_confidence
        self.min_track_conf = min_tracking_confidence
        self.max_num_hands = max_num_hands
        self.imgsz = imgsz
        self.skip_frames = skip_frames 

        self.running = False
        self.hands_model = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.recording = False
        self.source_fps: float = 30.0

    def _convert_to_qimage(self, frame: np.ndarray) -> QImage:
        if frame is None: return QImage()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

    def _classify_gesture(self, landmarks, handedness: str) -> str:
        # Улучшенная логика определения жестов
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        extended = []
        for tip, pip in zip(tips, pips):
            extended.append(landmarks[tip].y < landmarks[pip].y)
        
        is_right = handedness == "Right"
        thumb_tip, thumb_ip = 4, 3
        if is_right:
            thumb_ext = landmarks[thumb_tip].x < landmarks[thumb_ip].x
        else:
            thumb_ext = landmarks[thumb_tip].x > landmarks[thumb_ip].x
            
        extended.insert(0, thumb_ext)
        count = sum(extended)
        
        if count == 0: return "Fist"
        if count == 5: return "Open Palm"
        if extended[0] and not any(extended[1:]): return "Thumbs Up"
        if not extended[0] and extended[1] and extended[2] and not any(extended[3:]): return "Peace"
        if not extended[0] and extended[1] and not any(extended[2:]): return "Pointing"
        return f"Gest ({count})"

    def run(self) -> None:
        if not MEDIAPIPE_AVAILABLE:
            self.error_occurred.emit("MediaPipe not available.")
            return
        self.running = True
        try:
            # Используем model_complexity=0 для максимальной скорости на CPU
            self.hands_model = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_det_conf,
                min_tracking_confidence=self.min_track_conf,
                model_complexity=0 
            )
            logger.success("✅ MediaPipe Hands ready")
        except Exception as e:
            self.error_occurred.emit(f"Init error: {e}")
            return

        try:
            if self.source_type == "webcam": self._process_webcam()
            elif self.source_type == "video": self._process_video()
            elif self.source_type == "image": self._process_image()
        except Exception as e:
            self.error_occurred.emit(str(e))

        self._cleanup()

    def _process_webcam(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW для быстрого старта на Windows
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.source_fps = 30.0
        self._process_stream_loop()

    def _process_stream_loop(self):
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret: break
            
            frame_count += 1
            # Обрабатываем каждый кадр (или через один), если skip_frames > 1
            # Но отрисовку делаем всегда стабильной
            latency, hands_count, data, out_frame = self._run_inference(frame)
            
            q_img = self._convert_to_qimage(out_frame)
            if not q_img.isNull():
                self.frame_ready.emit(q_img)
            
            fps = frame_count / (time.time() - start_time)
            self.stats_ready.emit(fps, latency, hands_count)
            self.data_ready.emit(data)

    def _run_inference(self, frame: np.ndarray) -> Tuple[float, int, List[Dict], np.ndarray]:
        t0 = time.time()
        h, w = frame.shape[:2]
        # MediaPipe требует RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_model.process(rgb)
        latency = (time.time() - t0) * 1000
        
        out_frame = frame.copy()
        data_list = []
        hands_count = 0

        if results.multi_hand_landmarks:
            hands_count = len(results.multi_hand_landmarks)
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                conf = results.multi_handedness[idx].classification[0].score
                
                gesture = self._classify_gesture(hand_landmarks.landmark, handedness)
                
                # Рисуем скелет ВСЕГДА, когда есть результат
                mp_drawing.draw_landmarks(
                    out_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Ограничивающая рамка
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                x1, y1 = int(min(xs) * w), int(min(ys) * h)
                x2, y2 = int(max(xs) * w), int(max(ys) * h)
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(out_frame, f"{handedness}: {gesture}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                data_list.append({
                    "hand_id": idx,
                    "gesture": gesture,
                    "confidence": float(conf)
                })

        return latency, hands_count, data_list, out_frame

    def _cleanup(self):
        if self.cap: self.cap.release()
        if self.hands_model: self.hands_model.close()

    def stop(self):
        self.running = False
        self.wait(1000)