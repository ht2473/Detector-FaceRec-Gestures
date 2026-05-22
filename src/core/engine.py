"""YOLO object detection engine — runs inference in a dedicated QThread."""
from __future__ import annotations

import pathlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
from loguru import logger
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from ultralytics import YOLO

BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_device(requested: str) -> str:
    """Resolve 'auto'/'cuda'/'' → actual device string."""
    if requested.lower() in ("auto", "cuda", ""):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested.lower()


def _frame_to_qimage(frame_bgr: Any) -> QImage:
    """Convert a BGR numpy frame to a QImage (deep-copied for thread safety)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


class DetectionEngine(QThread):
    """YOLO detection engine — emits frames, stats and raw detection dicts."""

    frame_ready   = pyqtSignal(QImage)
    stats_ready   = pyqtSignal(float, float, int)   # fps, latency_ms, count
    data_ready    = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        source_type: str = "webcam",
        source_path: str = "",
        model_name:  str = "yolo26s.pt",
        conf:  float = 0.25,
        iou:   float = 0.45,
        imgsz: int   = 640,
        device: str  = "auto",
        classes: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.source_type = source_type.strip().lower()
        self.source_path = source_path.strip()
        self.model_name  = model_name.strip()
        self.conf  = conf
        self.iou   = iou
        self.imgsz = imgsz
        self.device = device
        self.classes = classes or []

        self.running = False
        self._model:    Optional[YOLO]              = None
        self._cap:      Optional[cv2.VideoCapture]  = None
        self._writer:   Optional[cv2.VideoWriter]   = None
        self._recording = False
        self._target_class_ids: Optional[List[int]] = None
        self._source_fps: float = 30.0

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        self.running = True

        # ── resolve device ──────────────────────────────────────────────────
        actual_device = _resolve_device(self.device)
        self.device = actual_device  # store resolved value so use_half check works
        logger.info(f"🖥️  Device: {actual_device.upper()}")

        if actual_device == "cuda":
            torch.backends.cudnn.benchmark    = True
            torch.backends.cudnn.deterministic = False

        # ── load model ──────────────────────────────────────────────────────
        model_path = MODELS_DIR / self.model_name
        logger.info(f"🤖 Loading model: {model_path.name}")

        try:
            self._model = YOLO(str(model_path))
            self._model.to(actual_device)
            logger.success(f"✅ Model loaded on {actual_device.upper()}")
        except Exception as exc:
            logger.error(f"❌ Model load error: {exc}")
            self.error_occurred.emit(str(exc))
            self.running = False
            return

        # ── build class-id filter ───────────────────────────────────────────
        if self.classes:
            names_lower = {v.lower(): k for k, v in self._model.names.items()}
            self._target_class_ids = [
                names_lower[c.lower()]
                for c in self.classes
                if c.lower() in names_lower
            ]
            logger.info(f"🔍 Filtering {len(self._target_class_ids)} classes")

        # ── dispatch ────────────────────────────────────────────────────────
        try:
            dispatch = {
                "webcam": self._process_webcam,
                "video":  self._process_video,
                "image":  self._process_image,
            }
            handler = dispatch.get(self.source_type)
            if handler is None:
                self.error_occurred.emit(f"Unknown source type: {self.source_type!r}")
            else:
                handler()
        except Exception as exc:
            logger.error(f"❌ Processing error: {exc}")
            self.error_occurred.emit(str(exc))
        finally:
            self._cleanup()
            logger.info("⏹️  Engine thread finished")

    # ------------------------------------------------------------------ source handlers
    def _process_webcam(self) -> None:
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self.error_occurred.emit("Failed to open webcam")
            return
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._source_fps = fps if fps > 0 else 30.0
        logger.info(f"📹 Webcam opened ({self._source_fps:.1f} FPS)")
        self._stream_loop("webcam")

    def _process_video(self) -> None:
        if not self.source_path or not pathlib.Path(self.source_path).exists():
            self.error_occurred.emit(f"Video not found: {self.source_path}")
            return
        self._cap = cv2.VideoCapture(self.source_path)
        if not self._cap.isOpened():
            self.error_occurred.emit("Failed to open video file")
            return
        fps    = self._cap.get(cv2.CAP_PROP_FPS)
        frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w      = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._source_fps = fps if fps > 0 else 30.0
        dur = frames / self._source_fps
        logger.info(
            f"🎬 Video: {pathlib.Path(self.source_path).name} "
            f"({w}×{h}, {self._source_fps:.1f} FPS, {frames} frames, {dur:.1f}s)"
        )
        self._stream_loop("video")

    def _process_image(self) -> None:
        path = pathlib.Path(self.source_path)
        if not path.exists():
            self.error_occurred.emit(f"Image not found: {self.source_path}")
            return
        frame = cv2.imread(str(path))
        if frame is None:
            self.error_occurred.emit(f"Cannot read image: {self.source_path}")
            return
        h, w = frame.shape[:2]
        logger.info(f"🖼️  Image: {path.name} ({w}×{h}px)")
        latency, count, detections, out_frame = self._run_inference(frame)
        for d in detections:
            d["filename"] = path.name
            d["filepath"] = str(path)
        self.stats_ready.emit(0.0, latency, count)
        self.data_ready.emit(detections)
        self.frame_ready.emit(_frame_to_qimage(out_frame))
        time.sleep(0.1)

    # ------------------------------------------------------------------ stream loop
    def _stream_loop(self, source_name: str) -> None:
        frame_count   = 0
        start_time    = time.time()
        consec_fails  = 0
        frame_interval = 1.0 / self._source_fps if self._source_fps > 0 else 1 / 30

        while self.running:
            t_loop = time.perf_counter()

            ret, frame = self._cap.read()
            if not ret or frame is None:
                consec_fails += 1
                if consec_fails >= 10:
                    logger.info(f"🎬 End of stream ({source_name}), frames={frame_count}")
                    break
                time.sleep(0.005)
                continue

            consec_fails = 0
            if frame.shape[0] < 10 or frame.shape[1] < 10:
                continue

            frame_count += 1
            latency, count, detections, out_frame = self._run_inference(frame)

            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            self.stats_ready.emit(fps, latency, count)
            self.data_ready.emit(detections)
            self.frame_ready.emit(_frame_to_qimage(out_frame))

            # pace video files to original speed
            if self.source_type == "video":
                sleep = max(0.0, frame_interval - (time.perf_counter() - t_loop))
                if sleep > 0.002:
                    time.sleep(sleep)

    # ------------------------------------------------------------------ inference
    def _run_inference(
        self, frame: Any
    ) -> Tuple[float, int, List[Dict], Any]:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # down-scale imgsz for very large frames to keep latency predictable
        imgsz = min(self.imgsz, 640) if max(h, w) > 1920 else self.imgsz
        use_half = (
            self.device == "cuda"
            and torch.cuda.is_available()
            and self._model is not None
        )

        kwargs: Dict[str, Any] = {
            "conf":    self.conf,
            "iou":     self.iou,
            "imgsz":   imgsz,
            "verbose": False,
            "augment": False,
            "half":    use_half,
            "stream":  False,
        }
        if self._target_class_ids is not None:
            kwargs["classes"] = self._target_class_ids

        try:
            results = self._model(frame, **kwargs)
        except Exception as exc:
            logger.error(f"❌ Inference error: {exc}")
            return 0.0, 0, [], frame.copy()

        latency_ms   = (time.perf_counter() - t0) * 1000
        detections: List[Dict] = []
        out_frame    = frame.copy()
        total_count  = 0

        for result in results:
            if result.boxes is None:
                continue
            total_count += len(result.boxes)
            try:
                out_frame = result.plot()
            except Exception as exc:
                logger.warning(f"⚠️  Draw error: {exc}")

            for box in result.boxes:
                try:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    xc, yc, bw, bh = box.xywh[0].tolist()
                    detections.append({
                        "class_id":   cls_id,
                        "class_name": self._model.names.get(cls_id, f"cls_{cls_id}"),
                        "confidence": round(conf, 4),
                        "bbox_xywh":  [round(v, 1) for v in (xc, yc, bw, bh)],
                        "timestamp":  datetime.now().isoformat(timespec="milliseconds"),
                        "frame_size": f"{w}x{h}",
                    })
                except Exception:
                    continue

        # ── overlay stats ──────────────────────────────────────────────────
        cv2.putText(out_frame, f"Latency: {latency_ms:.1f}ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(out_frame, f"Objects: {total_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # ── optional recording ─────────────────────────────────────────────
        if self._recording and self.source_type != "image":
            self._write_frame(out_frame)

        return latency_ms, total_count, detections, out_frame

    # ------------------------------------------------------------------ recording
    def _write_frame(self, frame: Any) -> None:
        try:
            if self._writer is None:
                h, w = frame.shape[:2]
                name = f"rec_{datetime.now():%H%M%S}.mp4"
                self._writer = cv2.VideoWriter(
                    str(OUTPUT_DIR / name),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(self._source_fps, 1.0),
                    (w, h),
                )
            self._writer.write(frame)
        except Exception as exc:
            logger.error(f"❌ Recording error: {exc}")
            self._recording = False

    def toggle_recording(self) -> bool:
        """Toggle recording on/off.  Returns True when recording has just **stopped**."""
        self._recording = not self._recording
        if not self._recording and self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.success(f"💾 Video saved → {OUTPUT_DIR}")
            return True          # stopped
        return False             # started

    # ------------------------------------------------------------------ cleanup / stop
    def _cleanup(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._writer:
            self._writer.release()
            self._writer = None

    def stop(self) -> None:
        self.running = False
        self.wait(3000)
