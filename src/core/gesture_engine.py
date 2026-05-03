"""Gesture Recognition Engine based on MediaPipe Hands (Tasks API - Production Ready)"""
import cv2
import time
import pathlib
import urllib.request
import numpy as np
from typing import Optional, Dict, List, Tuple
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from loguru import logger

# Инициализация MediaPipe
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe.framework.formats import landmark_pb2
    
    mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    MEDIAPIPE_AVAILABLE = True
    logger.success("✅ MediaPipe Tasks API ready")
except Exception as e:
    logger.error(f"❌ Initialization error: {e}")

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
TASK_FILE_PATH = BASE_DIR / "hand_landmarker.task"

class GestureRecognitionEngine(QThread):
    # Сигналы для UI
    frame_ready = pyqtSignal(QImage)
    stats_ready = pyqtSignal(float, float, int)  # fps, latency, hands_count
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
        skip_frames: int = 1
    ):
        super().__init__()
        self.source_type = source_type.strip().lower()
        self.source_path = source_path.strip()
        self.device = device.strip().lower()
        self.min_det_conf = min_detection_confidence
        self.min_track_conf = min_tracking_confidence
        self.max_num_hands = max_num_hands
        
        self.running = False
        self.hands_model = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.start_time: float = 0.0

    def _download_model_if_needed(self):
        """Гарантирует наличие файла модели"""
        if not TASK_FILE_PATH.exists():
            logger.info("Downloading hand_landmarker.task...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, TASK_FILE_PATH)

    def _convert_to_qimage(self, frame: np.ndarray) -> QImage:
        if frame is None: return QImage()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

    def _classify_gesture(self, landmarks, handedness: str) -> str:
        """Базовая логика определения жестов по точкам"""
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        extended = [landmarks[tip].y < landmarks[pip].y for tip, pip in zip(tips, pips)]
        count = sum(extended)
        if count == 0: return "Fist"
        if count == 4: return "Open Palm"
        return f"Gest ({count})"

    def run(self) -> None:
        if not MEDIAPIPE_AVAILABLE:
            self.error_occurred.emit("MediaPipe not available.")
            return
        
        try:
            self._download_model_if_needed()
            delegate = python.BaseOptions.Delegate.GPU if self.device == "gpu" else python.BaseOptions.Delegate.CPU
            
            # Настройка режима работы модели
            mode = vision.RunningMode.IMAGE if self.source_type == "image" else vision.RunningMode.VIDEO

            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(TASK_FILE_PATH), delegate=delegate),
                running_mode=mode,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_det_conf,
                min_hand_presence_confidence=self.min_track_conf,
                min_tracking_confidence=self.min_track_conf
            )
            self.hands_model = vision.HandLandmarker.create_from_options(options)
            logger.info(f"Engine started on {self.device.upper()} ({self.source_type})")
        except Exception as e:
            self.error_occurred.emit(f"Init error: {e}")
            return

        self.running = True
        try:
            if self.source_type == "image":
                self._process_image()
            elif self.source_type in ["video", "video file"]:
                self._process_video()
            else:
                self._process_webcam()
        except Exception as e:
            logger.exception("Runtime error in engine")
            self.error_occurred.emit(str(e))
        finally:
            self._cleanup()

    def _process_image(self):
        frame = cv2.imread(self.source_path)
        if frame is None:
            self.error_occurred.emit("Image file not found.")
            return
        latency, hands_count, data, out_frame = self._run_inference(frame)
        self.frame_ready.emit(self._convert_to_qimage(out_frame))
        self.stats_ready.emit(0.0, latency, hands_count)
        self.data_ready.emit(data)
        self.running = False

    def _process_video(self):
        self.cap = cv2.VideoCapture(self.source_path)
        self._process_stream_loop()

    def _process_webcam(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self._process_stream_loop()

    def _process_stream_loop(self):
        self.start_time = time.time()
        frame_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret: break
            
            frame_count += 1
            latency, hands_count, data, out_frame = self._run_inference(frame)
            
            # Отправляем кадр и метрики в UI
            self.frame_ready.emit(self._convert_to_qimage(out_frame))
            fps = frame_count / (time.time() - self.start_time)
            self.stats_ready.emit(fps, latency, hands_count)
            self.data_ready.emit(data)

    def _run_inference(self, frame: np.ndarray) -> Tuple[float, int, List[Dict], np.ndarray]:
        t0 = time.time()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Выбор метода детекции в зависимости от режима
        if self.source_type == "image":
            results = self.hands_model.detect(mp_image)
        else:
            ts = int((time.time() - self.start_time) * 1000)
            results = self.hands_model.detect_for_video(mp_image, ts)
        
        latency = (time.time() - t0) * 1000
        out_frame = frame.copy()
        data_list = []
        
        if results.hand_landmarks:
            for idx, landmarks in enumerate(results.hand_landmarks):
                handedness = results.handedness[idx][0].category_name
                conf = results.handedness[idx][0].score
                gesture = self._classify_gesture(landmarks, handedness)
                
                # Отрисовка скелета
                proto = landmark_pb2.NormalizedLandmarkList()
                proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in landmarks])
                mp_drawing.draw_landmarks(
                    out_frame, proto, mp_hands_connections, 
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Отрисовка текста и рамки
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                x1, y1 = int(min(xs) * w), int(min(ys) * h)
                cv2.putText(out_frame, f"{handedness}: {gesture}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                data_list.append({"hand_id": idx, "gesture": gesture, "confidence": float(conf)})
                
        return latency, len(results.hand_landmarks) if results.hand_landmarks else 0, data_list, out_frame

    def _cleanup(self):
        if self.cap: self.cap.release()
        if self.hands_model: self.hands_model.close()
        logger.info("Engine resources cleaned up")

    def stop(self):
        self.running = False
        self.wait(2000)