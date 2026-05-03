"""Detection engine for YOLO26 model inference"""
import cv2
import time
import pathlib
import torch
from datetime import datetime
from typing import Optional, List, Dict, Any
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from ultralytics import YOLO
from loguru import logger

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class DetectionEngine(QThread):
    """Main detection engine running in separate thread"""
    frame_ready = pyqtSignal(QImage)
    stats_ready = pyqtSignal(float, float, int)  # fps, latency_ms, object_count
    data_ready = pyqtSignal(list)  # List of detected objects
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        source_type: str = "webcam",
        source_path: str = "",
        model_name: str = "yolo26s.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cuda",
        classes: Optional[List[str]] = None
    ):
        super().__init__()
        self.source_type = source_type.strip()
        self.source_path = source_path.strip()
        self.model_name = model_name.strip()
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device.strip()
        self.classes = classes

        self.running = False
        self.model: Optional[YOLO] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.recording = False
        self.target_classes_idx: Optional[List[int]] = None
        self.source_fps: float = 30.0

    def run(self) -> None:
        self.running = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        model_path = MODELS_DIR / self.model_name
        logger.info(f"🤖 Loading model: {model_path.name}")

        try:
            if not model_path.exists():
                logger.warning(f"Model file not found locally: {model_path}. Trying auto-download...")

            self.model = YOLO(str(model_path))

            if self.device in ("auto", "cuda"):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"✅ Device selected: {self.device.upper()}")
            self.model.to(self.device)

            if self.classes:
                self.target_classes_idx = [
                    idx for name, idx in self.model.names.items()
                    if name.lower() in [c.lower() for c in self.classes]
                ]
                logger.info("🔍 Class filter applied")

        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            self.error_occurred.emit(str(e))
            self.running = False
            return

        try:
            if self.source_type == "webcam":
                self._process_webcam()
            elif self.source_type == "video":
                self._process_video()
            elif self.source_type == "image":
                self._process_image()
            else:
                self.error_occurred.emit(f"Unknown source type: {self.source_type}")
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            self.error_occurred.emit(str(e))

        self._cleanup()
        logger.info("⏹️ Engine thread finished naturally")

    def _process_webcam(self) -> None:
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.error_occurred.emit("Failed to open webcam")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.source_fps <= 0:
            self.source_fps = 30.0

        logger.info(f"📹 Webcam started ({self.source_fps:.1f} FPS)")
        self._process_stream_loop("webcam")

    def _process_video(self) -> None:
        if not self.source_path or not pathlib.Path(self.source_path).exists():
            self.error_occurred.emit(f"Video not found: {self.source_path}")
            return

        try:
            self.cap = cv2.VideoCapture(self.source_path)
            if not self.cap.isOpened():
                self.error_occurred.emit("Failed to open video file")
                return

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / self.source_fps if self.source_fps > 0 else 0

            logger.info(f"🎬 Video: {pathlib.Path(self.source_path).name}")
            logger.info(f"   Resolution: {width}x{height}, {self.source_fps:.2f} FPS, {total_frames} frames, {duration:.1f}s")
        except Exception as e:
            logger.error(f"❌ Video initialization error: {e}")
            self.error_occurred.emit(f"Video error: {str(e)}")
            if self.cap:
                self.cap.release()
            return

        self._process_stream_loop("video")

    def _process_stream_loop(self, source_name: str) -> None:
        frame_count = 0
        start_time = time.time()
        consecutive_failures = 0
        max_failures = 10

        while self.running:
            t_loop = time.time()
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.info(f"🎬 End of stream ({source_name}). Processed frames: {frame_count}")
                        break
                    time.sleep(0.005)
                    continue

                consecutive_failures = 0
                frame_count += 1

                if frame.shape[0] < 10 or frame.shape[1] < 10:
                    continue

                latency, count, frame_data = self._run_inference(frame, source_name)
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                self.stats_ready.emit(fps, latency, count)
                self.data_ready.emit(frame_data)

                if self.source_type == "video" and self.source_fps > 0:
                    target_frame_time = 1.0 / self.source_fps
                    processing_time = time.time() - t_loop
                    sleep_time = max(0.0, target_frame_time - processing_time)
                    if sleep_time > 0.002:
                        time.sleep(sleep_time)
                else:
                    time.sleep(0.001)

            except Exception as e:
                logger.error(f"❌ Critical error in loop: {e}")
                break

    def _process_image(self) -> None:
        if not self.source_path or not pathlib.Path(self.source_path).exists():
            self.error_occurred.emit(f"Image not found: {self.source_path}")
            return

        try:
            frame = cv2.imread(self.source_path)
            if frame is None:
                self.error_occurred.emit(f"Failed to read image: {self.source_path}")
                return

            h, w = frame.shape[:2]
            logger.info(f"🖼️ Processing image: {w}x{h}px")
            latency, count, frame_data = self._run_inference(frame, "image")

            for item in frame_data:
                item['filename'] = pathlib.Path(self.source_path).name
                item['filepath'] = str(self.source_path)

            self.stats_ready.emit(0.0, latency, count)
            self.data_ready.emit(frame_data)
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"❌ Image processing error: {e}")
            self.error_occurred.emit(f"Processing error: {str(e)}")

    def _run_inference(self, frame: Any, source_type: str = "stream") -> tuple:
        t0 = time.time()
        h, w = frame.shape[:2]

        try:
            target_size = self.imgsz
            if max(h, w) > 1920 and target_size > 640:
                target_size = min(target_size, 640)

            kwargs = {
                "conf": self.conf,
                "iou": self.iou,
                "imgsz": target_size,
                "verbose": False,
                "augment": False,
                "half": self.device == "cuda" and torch.cuda.is_available(),
                "stream": False
            }
            if self.target_classes_idx is not None:
                kwargs["classes"] = self.target_classes_idx

            results = self.model(frame, **kwargs)
        except Exception as e:
            logger.error(f"❌ Inference error: {e}")
            return 0, 0, []

        latency = (time.time() - t0) * 1000
        detected_count = 0
        detection_list = []
        output_frame = frame.copy()

        for result in results:
            if result.boxes is None:
                continue
            detected_count += len(result.boxes)

            try:
                output_frame = result.plot()
            except Exception as e:
                logger.warning(f"⚠️ Failed to draw boxes: {e}")

            for box in result.boxes:
                try:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xywh[0].tolist()
                    x, y, bw, bh = bbox
                    bbox = [max(0, min(x, w)), max(0, min(y, h)), min(bw, w), min(bh, h)]

                    detection_list.append({
                        "class_id": cls_id,
                        "class_name": self.model.names.get(cls_id, f"class_{cls_id}"),
                        "confidence": conf,
                        "bbox": bbox,
                        "timestamp": datetime.now().isoformat(),
                        "frame_size": f"{w}x{h}"
                    })
                except Exception:
                    continue

        if self.recording and source_type != "image":
            try:
                if self.writer is None:
                    h_frame, w_frame, _ = output_frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_name = f"rec_{datetime.now().strftime('%H%M%S')}.mp4"
                    fps = self.source_fps if self.source_fps > 0 else 30.0
                    self.writer = cv2.VideoWriter(str(OUTPUT_DIR / out_name), fourcc, fps, (w_frame, h_frame))
                self.writer.write(output_frame)
            except Exception as e:
                logger.error(f"❌ Video recording error: {e}")
                self.recording = False

        try:
            cv2.putText(output_frame, f"Latency: {latency:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Objects: {detected_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if len(output_frame.shape) == 3:
                rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                h_frame, w_frame, ch = rgb.shape
                q_img = QImage(rgb.data, w_frame, h_frame, ch * w_frame, QImage.Format.Format_RGB888).copy()
                self.frame_ready.emit(q_img)
        except Exception as e:
            logger.error(f"❌ Frame conversion error: {e}")

        return latency, detected_count, detection_list

    def toggle_recording(self) -> bool:
        self.recording = not self.recording
        if not self.recording and self.writer:
            self.writer.release()
            self.writer = None
            logger.success(f"💾 Video saved to: {OUTPUT_DIR}")
            return True
        return False

    def _cleanup(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.writer:
            self.writer.release()
            self.writer = None

    def stop(self) -> None:
        self.running = False
        self.wait(2000)