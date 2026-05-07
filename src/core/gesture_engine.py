"""Gesture Recognition Engine based on MediaPipe Hands (Tasks API - Production Ready)"""
import cv2
import time
import sys
import math
import pathlib
import urllib.request
import threading
import numpy as np
from typing import Optional, Dict, List, Tuple
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from loguru import logger

# Инициализация MediaPipe
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    from mediapipe.framework.formats import landmark_pb2
    
    mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    MEDIAPIPE_AVAILABLE = True
    logger.success("✅ MediaPipe Tasks API ready")
except ImportError as e:
    logger.error(f"❌ MediaPipe not installed: {e}")
except Exception as e:
    logger.error(f"❌ Initialization error: {e}")

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
BASE_DIR.mkdir(parents=True, exist_ok=True) 

TASK_FILE_PATH = BASE_DIR / "hand_landmarker.task"
OUTPUT_DIR = BASE_DIR / "output"

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
        self.last_ts: int = -1  
        
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.record_lock = threading.Lock() # Защита записи видео от состояния гонки

    def _download_model_if_needed(self):
        """Гарантирует наличие файла модели с обработкой ошибок сети"""
        if not TASK_FILE_PATH.exists():
            logger.info("Downloading hand_landmarker.task...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(url, TASK_FILE_PATH)
                logger.success("Model downloaded successfully.")
            except urllib.error.URLError as e:
                logger.error(f"Network error while downloading model: {e}")
                raise RuntimeError("Failed to download MediaPipe model. Check internet connection.")
            except Exception as e:
                logger.error(f"Unexpected error downloading model: {e}")
                raise e

    def _convert_to_qimage(self, frame: np.ndarray) -> QImage:
        """Безопасная конвертация OpenCV (BGR) -> PyQt (QImage RGB)"""
        if frame is None or frame.size == 0: 
            return QImage()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # bytesPerLine (ch * w) критически важен, иначе изображение "съедет" по диагонали
        return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

    def _classify_gesture(self, landmarks, handedness: str) -> str:
        """Базовая эвристика для классификации жестов"""
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        # Простая проверка: если кончик пальца выше сустава (Y меньше)
        extended = [landmarks[tip].y < landmarks[pip].y for tip, pip in zip(tips, pips)]
        count = sum(extended)
        
        if count == 0: return "Fist"
        if count == 4: return "Open Palm"
        return f"Gest ({count})"

    def run(self) -> None:
        """Основной метод потока QThread"""
        if not MEDIAPIPE_AVAILABLE:
            self.error_occurred.emit("MediaPipe not available. Check dependencies.")
            return
        
        try:
            self._download_model_if_needed()
            delegate = mp.tasks.BaseOptions.Delegate.GPU if self.device == "gpu" else mp.tasks.BaseOptions.Delegate.CPU
            
            mode = mp.tasks.vision.RunningMode.IMAGE if self.source_type == "image" else mp.tasks.vision.RunningMode.VIDEO

            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(TASK_FILE_PATH), delegate=delegate),
                running_mode=mode,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_det_conf,
                min_hand_presence_confidence=self.min_track_conf,
                min_tracking_confidence=self.min_track_conf
            )
            self.hands_model = mp.tasks.vision.HandLandmarker.create_from_options(options)
            logger.info(f"Engine started on {self.device.upper()} | Mode: {self.source_type}")
            
        except Exception as e:
            self.error_occurred.emit(f"Model Initialization error: {e}")
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
            logger.exception("Runtime error in engine loop")
            self.error_occurred.emit(f"Runtime error: {e}")
        finally:
            self._cleanup()

    def _process_image(self):
        frame = cv2.imread(self.source_path)
        if frame is None:
            self.error_occurred.emit(f"Failed to load image: {self.source_path}")
            return
        latency, hands_count, data, out_frame = self._run_inference(frame)
        self.frame_ready.emit(self._convert_to_qimage(out_frame))
        self.stats_ready.emit(0.0, latency, hands_count)
        self.data_ready.emit(data)
        self.running = False

    def _process_video(self):
        self.cap = cv2.VideoCapture(self.source_path)
        if not self.cap.isOpened():
            self.error_occurred.emit(f"Could not open video file: {self.source_path}")
            return
        self._process_stream_loop(is_video_file=True)

    def _process_webcam(self):
        # Оптимизированный запуск вебкамеры (DSHOW работает быстрее на Windows)
        if sys.platform.startswith('win'):
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(0)
            
        if not self.cap.isOpened():
            self.error_occurred.emit("Could not open webcam.")
            return
        
        # Запрашиваем у камеры стандартное разрешение (опционально)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self._process_stream_loop(is_video_file=False)

    def _process_stream_loop(self, is_video_file: bool):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Получаем реальный FPS источника
        source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or math.isnan(source_fps) or source_fps <= 0:
            source_fps = 30.0  # Fallback если камера/видео не отдают FPS
            
        frame_delay_sec = 1.0 / source_fps
        self.start_time = time.perf_counter()
        self.last_ts = -1
        frame_count = 0
        
        while self.running:
            loop_start = time.perf_counter()
            
            ret, frame = self.cap.read()
            if not ret: 
                break # Конец видео или камера отвалилась
            
            frame_count += 1
            
            # --- Расчет Timestamp для MediaPipe ---
            if is_video_file:
                # Для файлов берем точное время прямо из видеопотока
                raw_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                current_ts = int(raw_msec) if raw_msec >= 0 else self.last_ts + 1
            else:
                # Для вебкамеры считаем реальное время работы
                current_ts = int((time.perf_counter() - self.start_time) * 1000)

            # Гарантия строгого возрастания (требование Tasks API)
            if current_ts <= self.last_ts:
                current_ts = self.last_ts + 1
            self.last_ts = current_ts

            # --- Инференс ---
            latency, hands_count, data, out_frame = self._run_inference(frame, current_ts)
            
            # --- Запись Видео ---
            with self.record_lock:
                if self.is_recording:
                    if self.video_writer is None:
                        path = OUTPUT_DIR / f"gesture_rec_{int(time.time())}.mp4"
                        h, w = out_frame.shape[:2]
                        # ИСПОЛЬЗУЕМ source_fps вместо хардкода 30, чтобы видео не ускорялось!
                        self.video_writer = cv2.VideoWriter(
                            str(path), 
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            source_fps, 
                            (w, h)
                        )
                    self.video_writer.write(out_frame)
            
            # --- Отправка в UI ---
            self.frame_ready.emit(self._convert_to_qimage(out_frame))
            fps_current = frame_count / (time.perf_counter() - self.start_time)
            self.stats_ready.emit(fps_current, latency, hands_count)
            self.data_ready.emit(data)

            # --- Синхронизация FPS для видеофайлов ---
            # Усыпляем поток, если обработка кадра заняла меньше времени, чем нужно для нормального FPS
            if is_video_file:
                process_time = time.perf_counter() - loop_start
                sleep_time = frame_delay_sec - process_time
                if sleep_time > 0:
                    self.msleep(int(sleep_time * 1000))

    def _run_inference(self, frame: np.ndarray, timestamp_ms: int = 0) -> Tuple[float, int, List[Dict], np.ndarray]:
        t0 = time.perf_counter()
        
        # MediaPipe работает с RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        if self.source_type == "image":
            results = self.hands_model.detect(mp_image)
        else:
            results = self.hands_model.detect_for_video(mp_image, timestamp_ms)
        
        latency = (time.perf_counter() - t0) * 1000  # в миллисекундах
        
        out_frame = frame.copy()
        data_list = []
        
        if results.hand_landmarks:
            h, w = frame.shape[:2]
            for idx, landmarks in enumerate(results.hand_landmarks):
                handedness = results.handedness[idx][0].category_name
                conf = results.handedness[idx][0].score
                gesture = self._classify_gesture(landmarks, handedness)
                
                # Отрисовка скелета
                proto = landmark_pb2.NormalizedLandmarkList()
                proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in landmarks
                ])
                mp_drawing.draw_landmarks(
                    out_frame, proto, mp_hands_connections, 
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Вычисляем bounding box для текста
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                x1, y1 = int(min(xs) * w), int(min(ys) * h)
                
                cv2.putText(
                    out_frame, f"{handedness}: {gesture}", 
                    (x1, max(y1 - 10, 20)), # max спасает текст от выхода за верхнюю границу экрана
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

                data_list.append({
                    "hand_id": idx, 
                    "handedness": handedness, 
                    "gesture": gesture, 
                    "confidence": float(conf)
                })
                
        return latency, len(results.hand_landmarks) if results.hand_landmarks else 0, data_list, out_frame

    def toggle_recording(self) -> bool:
        """Потокобезопасное переключение записи"""
        with self.record_lock:
            self.is_recording = not self.is_recording
            if not self.is_recording and self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
        return self.is_recording

    def _cleanup(self):
        """Гарантированная очистка ресурсов"""
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
        
        with self.record_lock:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                
        if self.hands_model is not None:
            self.hands_model.close()
            
        logger.info("✅ Engine resources cleaned up")

    def stop(self):
        """Метод вызывается из UI для остановки потока"""
        self.running = False
        self.wait(2000) # Даем потоку 2 секунды на мягкое завершение