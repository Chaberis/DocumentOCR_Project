from pathlib import Path

# --- Пути ---
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / 'models'
DETECTION_MODEL_PATH = MODELS_DIR / 'best.pt'
OCR_MODEL_PATH = MODELS_DIR / 'trocr-base-handwritten-ru'
FONT_PATH = ROOT_DIR / 'assets' / 'ARIAL.TTF'

# --- Hugging Face ---
DETECTION_REPO_ID = "Chaberis/docu_scribe_yolov12m"
DETECTION_FILENAME = "best.pt"
OCR_REPO_ID = 'kazars24/trocr-base-handwritten-ru'

# --- Настройки ---
PDF_DPI = 200
CONFIDENCE = 0.2