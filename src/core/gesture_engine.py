"""Gesture Recognition Engine based on Metaidigitcv (cvzone wrapper) - Fixed Playback & Recording"""
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
    from cvzone.HandTrackingModule import HandDetector
    CVZONE_AVAILABLE = True
except ImportError:
    CVZONE_AVAILABLE = False

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
        
        self.running = False
        self.detector = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.record_lock = threading.Lock()

    def _convert_to_qimage(self, frame: np.ndarray) -> QImage:
        if frame is None or frame.size == 0:
            return QImage()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # Важно: bytesPerLine = ch * w для предотвращения искажений изображения
        return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

    def run(self) -> None:
        if not CVZONE_AVAILABLE:
            self.error_occurred.emit("Install dependencies: pip install cvzone mediapipe")
            return

        try:
            self.detector = HandDetector(detectionCon=self.min_det_conf, maxHands=self.max_hands)
            logger.info(f"Engine Metaidigitcv started | Mode: {self.source_type}")
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
        
        # Подключаемся к камере или открываем файл
        self.cap = cv2.VideoCapture(self.source_path if is_video else 0)
        
        if not self.cap.isOpened():
            self.error_occurred.emit("Failed to open video source")
            return

        if not is_video:
            # Настройки для веб-камеры: высокое разрешение, минимальный буфер
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Безопасное получение FPS
        fps_source = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps_source or fps_source <= 0 or math.isnan(fps_source):
            fps_source = 30.0  # Fallback если камера/файл не отдает FPS
            
        frame_delay = 1.0 / fps_source  # Время, которое должен занимать 1 кадр

        start_time = time.time()
        frame_count = 0

        while self.running:
            loop_start = time.perf_counter() # Засекаем начало обработки кадра

            ret, frame = self.cap.read()
            if not ret:
                break # Конец файла видео или отключение камеры

            frame_count += 1
            latency, hands_count, data, out_frame = self._run_inference(frame)

            # Запись кадра (потокобезопасно)
            with self.record_lock:
                if self.is_recording:
                    if self.video_writer is None:
                        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                        path = OUTPUT_DIR / f"rec_{int(time.time())}.mp4"
                        h, w = out_frame.shape[:2]
                        # Используем fps_source для правильной скорости воспроизведения записанного видео
                        self.video_writer = cv2.VideoWriter(
                            str(path), 
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            fps_source, 
                            (w, h)
                        )
                    self.video_writer.write(out_frame)

            curr_fps = frame_count / (time.time() - start_time)
            
            # Отправка сигналов в GUI
            self.frame_ready.emit(self._convert_to_qimage(out_frame))
            self.stats_ready.emit(curr_fps, latency, hands_count)
            self.data_ready.emit(data)

            # --- ИСПРАВЛЕНИЕ: Искусственная задержка для видеофайлов ---
            if is_video:
                processing_time = time.perf_counter() - loop_start
                sleep_time = frame_delay - processing_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _run_inference(self, frame: np.ndarray) -> Tuple[float, int, List[Dict], np.ndarray]:
        t0 = time.perf_counter()
        
        # findHands возвращает список рук. flipType=True исправляет зеркальность
        hands, out_frame = self.detector.findHands(frame, draw=True, flipType=True)
        
        latency = (time.perf_counter() - t0) * 1000
        data_list = []

        for hand in hands:
            fingers = self.detector.fingersUp(hand)
            
            # Определяем жест по комбинации пальцев [Большой, Указательный, Средний, Безымянный, Мизинец]
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

            # БЕЗОПАСНОЕ извлечение данных
            score = hand.get("score", 1.0) 
            hand_type = hand.get("type", "Unknown")
            bbox = hand.get("bbox", [0, 0, 0, 0])

            # Отрисовка текста над рукой
            cv2.putText(out_frame, f"{hand_type}: {gesture}", (bbox[0] +70, bbox[1] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            data_list.append({
                "handedness": hand_type,
                "gesture": gesture,
                "confidence": score,
                "bbox": bbox
            })

        return latency, len(hands), data_list, out_frame

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
        logger.info("✅ Engine resources cleaned up")

    def stop(self):
        self.running = False
        self.wait(2000)