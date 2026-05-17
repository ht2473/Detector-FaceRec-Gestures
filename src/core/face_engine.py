"""InsightFace face-recognition engine — runs in a dedicated QThread (CPU-only)."""
from __future__ import annotations

import pathlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Image extensions accepted for the face database
_FACE_DB_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _frame_to_qimage(frame_bgr: np.ndarray) -> QImage:
    """BGR → QImage (deep copy for thread safety)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return QImage()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


class FaceRecognitionEngine(QThread):
    """Face detection + recognition engine based on InsightFace (buffalo_sc, CPU)."""

    frame_ready    = pyqtSignal(QImage)
    stats_ready    = pyqtSignal(float, float, int)   # fps, latency_ms, face_count
    data_ready     = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        source_type:  str   = "webcam",
        source_path:  str   = "",
        device:       str   = "cpu",       # InsightFace is always CPU here
        det_thresh:   float = 0.5,
        rec_thresh:   float = 0.35,
        face_db_dir:  str   = "",
        imgsz:        int   = 320,
        skip_frames:  int   = 3,
    ) -> None:
        super().__init__()
        self.source_type  = source_type.strip().lower()
        self.source_path  = source_path.strip()
        self.det_thresh   = det_thresh
        self.rec_thresh   = rec_thresh
        self.imgsz        = imgsz
        self.skip_frames  = max(1, skip_frames)
        self.face_db_dir  = pathlib.Path(face_db_dir) if face_db_dir else None

        self.running      = False
        self._app:        Optional[FaceAnalysis]      = None
        self._cap:        Optional[cv2.VideoCapture]  = None
        self._writer:     Optional[cv2.VideoWriter]   = None
        self._recording   = False
        self._source_fps: float = 30.0
        # name → L2-normalised embedding vector
        self._face_db:    Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        if not INSIGHTFACE_AVAILABLE:
            self.error_occurred.emit(
                "insightface is not installed. Run: pip install insightface onnxruntime"
            )
            return

        self.running = True

        try:
            logger.info("🤖 Initialising InsightFace (buffalo_sc) on CPU …")
            self._app = FaceAnalysis(
                name="buffalo_sc",
                providers=["CPUExecutionProvider"],
            )
            # ctx_id=-1  →  CPU; det_size must be a tuple of multiples-of-32
            det_sz = (self.imgsz, self.imgsz)
            self._app.prepare(ctx_id=-1, det_thresh=self.det_thresh, det_size=det_sz)
            logger.success("✅ InsightFace ready")
            self._load_face_db()
        except Exception as exc:
            logger.error(f"❌ InsightFace init: {exc}")
            self.error_occurred.emit(str(exc))
            self.running = False
            return

        try:
            dispatch = {
                "webcam": self._process_webcam,
                "video":  self._process_video,
                "image":  self._process_image,
            }
            handler = dispatch.get(self.source_type)
            if handler is None:
                self.error_occurred.emit(f"Unknown source: {self.source_type!r}")
            else:
                handler()
        except Exception as exc:
            logger.error(f"❌ Processing error: {exc}")
            self.error_occurred.emit(str(exc))
        finally:
            self._cleanup()
            logger.info("⏹️  Face engine finished")

    # ------------------------------------------------------------------ face DB
    def _load_face_db(self) -> None:
        if not self.face_db_dir or not self.face_db_dir.is_dir():
            logger.warning("⚠️  No face DB directory — all faces will be 'Unknown'")
            return

        logger.info(f"📂 Loading face DB from: {self.face_db_dir}")
        loaded = 0
        for img_path in self.face_db_dir.iterdir():
            if img_path.suffix.lower() not in _FACE_DB_EXTS:
                continue
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                faces = self._app.get(img)
                if not faces:
                    logger.warning(f"⚠️  No face found in '{img_path.name}'")
                    continue
                emb = faces[0].embedding
                self._face_db[img_path.stem] = self._normalise(emb)
                loaded += 1
                logger.success(f"✅ Registered: {img_path.stem}")
            except Exception as exc:
                logger.warning(f"⚠️  Skipping '{img_path.name}': {exc}")

        logger.info(f"📚 Face DB ready: {loaded} identities")

    @staticmethod
    def _normalise(emb: np.ndarray) -> np.ndarray:
        """L2-normalise an embedding vector (required for cosine similarity via dot)."""
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8)

    def _match_face(self, emb: np.ndarray) -> Tuple[str, float]:
        """Return (name, cosine_similarity) for the best DB match."""
        if not self._face_db:
            return "Unknown", 0.0
        query = self._normalise(emb)
        names = list(self._face_db.keys())
        matrix = np.vstack(list(self._face_db.values()))   # (N, D)
        sims   = matrix @ query                              # cosine sims
        best   = int(np.argmax(sims))
        best_sim = float(sims[best])
        if best_sim >= self.rec_thresh:
            return names[best], best_sim
        return "Unknown", best_sim

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
        self._stream_loop()

    def _process_video(self) -> None:
        if not self.source_path or not pathlib.Path(self.source_path).exists():
            self.error_occurred.emit(f"Video not found: {self.source_path}")
            return
        self._cap = cv2.VideoCapture(self.source_path)
        if not self._cap.isOpened():
            self.error_occurred.emit("Failed to open video file")
            return
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._source_fps = fps if fps > 0 else 30.0
        logger.info(f"🎬 Video: {pathlib.Path(self.source_path).name}")
        self._stream_loop()

    def _process_image(self) -> None:
        path = pathlib.Path(self.source_path)
        if not path.exists():
            self.error_occurred.emit(f"Image not found: {self.source_path}")
            return
        frame = cv2.imread(str(path))
        if frame is None:
            self.error_occurred.emit("Failed to read image")
            return
        latency, count, data = self._run_inference(frame)
        out = self._draw_overlays(frame, data, latency, count)
        for d in data:
            d["filename"] = path.name
            d["filepath"] = str(path)
        self.stats_ready.emit(0.0, latency, count)
        self.data_ready.emit(data)
        self.frame_ready.emit(_frame_to_qimage(out))
        time.sleep(0.1)

    # ------------------------------------------------------------------ stream loop
    def _stream_loop(self) -> None:
        frame_count  = 0
        infer_count  = 0
        start_time   = time.time()
        fails        = 0
        frame_interval = 1.0 / self._source_fps

        # cached inference results for skipped frames
        last_latency  = 0.0
        last_count    = 0
        last_data: List[Dict] = []

        while self.running:
            t_loop = time.perf_counter()

            ret, frame = self._cap.read()
            if not ret or frame is None:
                fails += 1
                if fails >= 10:
                    break
                time.sleep(0.005)
                continue

            fails = 0
            frame_count += 1

            # run inference every N-th frame to stay real-time on CPU
            if infer_count % self.skip_frames == 0:
                last_latency, last_count, last_data = self._run_inference(frame)
            infer_count += 1

            out = self._draw_overlays(frame, last_data, last_latency, last_count)

            # recording (every frame for smooth output)
            if self._recording:
                self._write_frame(out)

            q_img = _frame_to_qimage(out)
            if not q_img.isNull():
                self.frame_ready.emit(q_img)

            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            self.stats_ready.emit(fps, last_latency, last_count)
            self.data_ready.emit(last_data)

            # pace video files
            if self.source_type == "video":
                sleep = max(0.0, frame_interval - (time.perf_counter() - t_loop))
                if sleep > 0.002:
                    time.sleep(sleep)

    # ------------------------------------------------------------------ inference
    def _run_inference(self, frame: np.ndarray) -> Tuple[float, int, List[Dict]]:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        faces = self._app.get(frame)
        latency_ms = (time.perf_counter() - t0) * 1000

        data: List[Dict] = []
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            name, sim = self._match_face(face.embedding)
            data.append({
                "name":        name,
                "similarity":  round(float(sim), 4),
                "det_score":   round(float(face.det_score), 4),
                "bbox":        [x1, y1, x2, y2],
                "timestamp":   datetime.now().isoformat(timespec="milliseconds"),
                "frame_size":  f"{w}x{h}",
            })

        return latency_ms, len(faces), data

    # ------------------------------------------------------------------ drawing
    def _draw_overlays(
        self,
        frame:      np.ndarray,
        data:       List[Dict],
        latency:    float,
        face_count: int,
    ) -> np.ndarray:
        out = frame.copy()
        for face in data:
            x1, y1, x2, y2 = face["bbox"]
            name   = face["name"]
            sim    = face["similarity"]
            color  = (0, 220, 0) if name != "Unknown" else (0, 60, 220)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{name}  {sim:.2f}"
            cv2.putText(out, label, (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        cv2.putText(out, f"Latency: {latency:.1f}ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(out, f"Faces: {face_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------ recording
    def _write_frame(self, frame: np.ndarray) -> None:
        try:
            if self._writer is None:
                h, w = frame.shape[:2]
                name = f"face_rec_{datetime.now():%H%M%S}.mp4"
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
        """Returns True when recording has just *stopped*."""
        self._recording = not self._recording
        if not self._recording and self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.success(f"💾 Face video saved → {OUTPUT_DIR}")
            return True
        return False

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
