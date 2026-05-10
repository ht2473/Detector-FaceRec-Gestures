"""Face Recognition Engine based on InsightFace (CPU Optimized)"""
import cv2
import time
import pathlib
import numpy as np
import onnxruntime as ort
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from insightface.app import FaceAnalysis
from loguru import logger

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class FaceRecognitionEngine(QThread):
    frame_ready = pyqtSignal(QImage)
    stats_ready = pyqtSignal(float, float, int)  # fps, latency_ms, face_count
    data_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        source_type: str = "webcam",
        source_path: str = "",
        device: str = "cpu",
        det_thresh: float = 0.5,
        rec_thresh: float = 0.35,
        face_db_dir: str = "",
        imgsz: int = 640,  
        skip_frames: int = 1 
    ):
        super().__init__()
        self.source_type = source_type.strip()
        self.source_path = source_path.strip()
        self.device = "cpu" 
        self.det_thresh = det_thresh
        self.rec_thresh = rec_thresh
        self.imgsz = imgsz
        self.skip_frames = max(1, skip_frames)
        self.face_db_dir = pathlib.Path(face_db_dir) if face_db_dir else None

        self.running = False
        self.app: Optional[FaceAnalysis] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.recording = False
        self.source_fps: float = 30.0
        self.face_db: Dict[str, np.ndarray] = {}

    def _convert_to_qimage(self, frame: np.ndarray) -> QImage:
        """Безопасная конвертация BGR кадра в QImage для PyQt6"""
        if frame is None:
            return QImage()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

    def _load_face_db(self):
        if not self.face_db_dir or not self.face_db_dir.is_dir():
            logger.warning("⚠️ Face DB directory not provided or invalid. All faces will be 'Unknown'.")
            return
        
        logger.info(f"📂 Loading face database from: {self.face_db_dir}")
        for img_path in self.face_db_dir.glob("*.[jp][pn]g"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                faces = self.app.get(img)
                if faces:
                    name = img_path.stem
                    self.face_db[name] = faces[0].embedding
                    logger.success(f"✅ Registered: {name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {img_path.name}: {e}")

    def run(self) -> None:
        self.running = True
        # 🔥 Принудительно используем CPU
        providers = ['CPUExecutionProvider']
        logger.info(f"🔍 ONNX Runtime providers: {providers}")

        try:
            logger.info("🤖 Initializing InsightFace (buffalo_sc) for CPU...")
            # 🔥 Модель buffalo_sc
            self.app = FaceAnalysis(name='buffalo_sc', providers=providers)
            # 🔥 ctx_id=-1 явно указывает InsightFace использовать CPU
            self.app.prepare(ctx_id=-1, det_thresh=self.det_thresh, det_size=(self.imgsz, self.imgsz))
            logger.success("✅ InsightFace initialized successfully on CPU")
            self._load_face_db()
        except Exception as e:
            logger.error(f"❌ InsightFace init error: {e}")
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
                self.error_occurred.emit(f"Unknown source: {self.source_type}")
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            self.error_occurred.emit(str(e))

        self._cleanup()
        logger.info("⏹️ Face engine finished")

    def _process_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.error_occurred.emit("Failed to open webcam")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._process_stream_loop("webcam")

    def _process_video(self):
        if not self.source_path or not pathlib.Path(self.source_path).exists():
            self.error_occurred.emit(f"Video not found: {self.source_path}")
            return
        self.cap = cv2.VideoCapture(self.source_path)
        if not self.cap.isOpened():
            self.error_occurred.emit("Failed to open video")
            return
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._process_stream_loop("video")

    def _process_stream_loop(self, source_name: str):
        frame_count = 0
        count = 0
        start_time = time.time()
        fails = 0

        # Кэш для пропущенных кадров
        last_data = []
        last_latency = 0.0
        last_face_count = 0

        while self.running:
            t_loop = time.time()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                fails += 1
                if fails >= 10:
                    break
                time.sleep(0.005)
                continue
            fails = 0
            frame_count += 1

            # 🔥 Логика пропуска кадров. Инференс только каждый N-ный кадр
            if count % self.skip_frames == 0:
                last_latency, last_face_count, last_data = self._run_inference(frame)
            
            count += 1

            # Рисуем данные ВСЕГДА на каждом кадре (исправляет мерцание)
            out_frame = self._draw_overlays(frame, last_data, last_latency, last_face_count)
            q_img = self._convert_to_qimage(out_frame)

            # Обработка записи видео для КАЖДОГО кадра (плавная запись)
            if self.recording:
                try:
                    if not self.writer:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        fps = self.source_fps if self.source_fps > 0 else 30.0
                        h, w = out_frame.shape[:2]
                        out_name = f"face_rec_{datetime.now():%H%M%S}.mp4"
                        self.writer = cv2.VideoWriter(str(OUTPUT_DIR / out_name), fourcc, fps, (w, h))
                    self.writer.write(out_frame)
                except Exception as e:
                    logger.error(f"❌ Recording error: {e}")
                    self.recording = False

            # Отправка сигналов в UI
            if not q_img.isNull():
                self.frame_ready.emit(q_img)
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            self.stats_ready.emit(fps, last_latency, last_face_count)
            self.data_ready.emit(last_data)

            # Синхронизация скорости для видеофайлов
            if self.source_type == "video" and self.source_fps > 0:
                target_frame_time = 1.0 / self.source_fps
                processing_time = time.time() - t_loop
                sleep_time = max(0.0, target_frame_time - processing_time)
                if sleep_time > 0.002:
                    time.sleep(sleep_time)
            else:
                time.sleep(0.001)

    def _draw_overlays(self, frame: np.ndarray, data: list, latency: float, face_count: int) -> np.ndarray:
        """Единый метод для отрисовки боксов, имен и статистики на кадре"""
        out_frame = frame.copy()
        
        # Отрисовка лиц
        for face in data:
            x1, y1, x2, y2 = face["bbox"]
            name = face["name"]
            sim = face["similarity"]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out_frame, f"{name} {sim:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # Отрисовка статистики всегда (решает проблему мерцания)
        cv2.putText(out_frame, f"Latency: {latency:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(out_frame, f"Faces: {face_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return out_frame

    def _process_image(self):
        if not self.source_path or not pathlib.Path(self.source_path).exists():
            self.error_occurred.emit(f"Image not found: {self.source_path}")
            return
        frame = cv2.imread(self.source_path)
        if frame is None:
            self.error_occurred.emit("Failed to read image")
            return
        
        latency, count, data = self._run_inference(frame)
        out_frame = self._draw_overlays(frame, data, latency, count)
        
        for d in data:
            d['filename'] = pathlib.Path(self.source_path).name
            d['filepath'] = str(self.source_path)
            
        self.stats_ready.emit(0.0, latency, count)
        self.data_ready.emit(data)
        self.frame_ready.emit(self._convert_to_qimage(out_frame))
        time.sleep(0.1)

    def _match_face(self, emb: np.ndarray) -> Tuple[str, float]:
        if not self.face_db:
            return "Unknown", 0.0
        
        names = list(self.face_db.keys())
        db_embs = np.vstack(list(self.face_db.values()))
        sims = db_embs @ emb
        best_idx = np.argmax(sims)
        best_sim = float(sims[best_idx])
        
        if best_sim >= self.rec_thresh:
            return names[best_idx], best_sim
        return "Unknown", best_sim

    def _run_inference(self, frame: np.ndarray) -> Tuple[float, int, List[Dict]]:
        """Только математика и вызов нейросети, без отрисовки"""
        t0 = time.time()
        h, w = frame.shape[:2]
        faces = self.app.get(frame)
        latency = (time.time() - t0) * 1000
        
        data_list = []
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            name, sim = self._match_face(face.embedding)

            data_list.append({
                "name": name,
                "similarity": float(sim),
                "det_score": float(face.det_score),
                "bbox": [x1, y1, x2, y2],
                "timestamp": datetime.now().isoformat(),
                "frame_size": f"{w}x{h}"
            })

        return latency, len(faces), data_list

    def toggle_recording(self) -> bool:
        self.recording = not self.recording
        if not self.recording and self.writer:
            self.writer.release()
            self.writer = None
            logger.success(f"💾 Face video saved to: {OUTPUT_DIR}")
            return True
        return False

    def _cleanup(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.writer:
            self.writer.release()
            self.writer = None

    def stop(self):
        self.running = False
        self.wait(2000)