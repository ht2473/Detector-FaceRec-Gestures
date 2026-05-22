# Detector + FaceRec + Gestures

Desktop application for real-time **object detection**, **face recognition** and **hand gesture recognition**, built with Python, PyQt6 and state-of-the-art ML models.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyQt6](https://img.shields.io/badge/PyQt6-6.7%2B-green)
![YOLO](https://img.shields.io/badge/YOLO-26-orange)
![Tests](https://img.shields.io/badge/tests-71%20passed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Features

| Module | Model | Backend |
|--------|-------|---------|
| Object Detection | YOLO 26 (n / s / m / l) | CPU or CUDA |
| Face Recognition | InsightFace `buffalo_sc` | CPU (ONNX Runtime) |
| Gesture Recognition | MediaPipe Hands | CPU |

- Live webcam, video file and static image as input sources
- Frame-skip optimisation for face recognition on CPU
- FP16 inference on CUDA for YOLO (automatic when GPU is available)
- Per-session JSON export of detections / faces / gestures
- Screenshot and video recording to `output/`
- Dark-themed PyQt6 UI with three independent tabs
- Configurable via `configs/default.yaml` — no code changes needed

---

## Requirements

- Python 3.11+
- GPU optional (CUDA 12+ recommended for YOLO acceleration)

```bash
pip install -r requirements.txt
```

> **Note** — PyTorch with CUDA must be installed separately if you need GPU support:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/ht2473/Detector-FaceRec-Gestures.git
cd Detector-FaceRec-Gestures

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place a YOLO model weights file in models/
#    e.g. models/yolo26s.pt — download from ultralytics releases

# 4. Run
python main.py
```

---

## Configuration

Application defaults are read from `configs/default.yaml` on startup — edit it to change confidence thresholds, model name, input size, skip-frames, and so on without touching the source code.

`configs/prompts.yaml` contains preset class lists (vehicles, animals, food, etc.) that can be used to filter YOLO detections.

---

## Face Recognition setup

1. Create a folder (e.g. `FaceDB/`) and put one reference photo per person inside.
   Name each file with the person's name: `Ivan_Petrov.jpg`, `Alice.png`, etc.
2. In the **Face Recognition** tab click **Load DB Folder…** and select your folder.
3. Start the engine — registered faces will be labelled with their names.

---

## Supported gestures

| Gesture | Fingers |
|---------|---------|
| Fist | all closed |
| Open Palm | all open |
| Pointing | index only |
| Victory / Peace | index + middle |
| Thumbs Up | thumb only |
| I Love You | thumb + index + pinky |
| Rock On | index + pinky |
| Call Me | thumb + pinky |
| Four | all except pinky |
| Four (no thumb) | index + middle + ring + pinky |
| Pinky | pinky only |

---

## Project structure

```
.
├── configs/
│   ├── default.yaml        # application defaults (read on startup)
│   └── prompts.yaml        # YOLO class presets
├── FaceDB/                 # reference face photos (gitignored)
├── logs/                   # runtime logs (gitignored)
├── models/                 # *.pt model weights (gitignored)
├── output/                 # recordings, screenshots, JSON (gitignored)
├── src/
│   ├── core/
│   │   ├── engine.py           # YOLO detection QThread
│   │   ├── face_engine.py      # InsightFace recognition QThread
│   │   └── gesture_engine.py   # MediaPipe gesture QThread
│   ├── gui/
│   │   └── window.py           # MainWindow (PyQt6)
│   └── utils/
│       └── helpers.py
├── tests/
│   ├── conftest.py             # stubs for heavy deps (no GPU/camera needed)
│   ├── test_engine.py
│   ├── test_face_engine.py
│   ├── test_gesture_engine.py
│   └── test_helpers.py
├── hand_landmarker.task    # MediaPipe model file (not tracked in git)
├── main.py
└── requirements.txt
```

---

## Running tests

Unit tests cover core logic (gesture classification, face matching, device resolution, config loading) and run without a camera, GPU, or any heavy ML library installed.

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## License

MIT
