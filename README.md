### Инструкция по запуску

```bash
# Клонирование репозитория
git clone https://github.com/ht2473/Detector-FaceRec-Gestures.git
cd Detector-FaceRec-Gestures

# Создание и активация виртуального окружения
python -m venv .venv (3.11.9)
.\(\venv\Scripts\Activate\)

# Обновление pip и установка зависимостей
python -m pip install --upgrade pip
pip install -r requirements.txt

# Ссылка на необходимые инструменты сборки C++ 
https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/

# Ручная установка библиотек
pip install onnxruntime
pip install insightface
pip install mediapipe
pip install cvzone mediapipe

# Запуск проекта
python main.py
```
