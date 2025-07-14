from pathlib import Path

# --- Пути ---
ROOT_DIR = Path(__file__).parent.parent
DETECTION_MODEL = ROOT_DIR / 'models/best.pt'
OCR_MODEL = ROOT_DIR / 'models/trocr-base-handwritten-ru'
FONT_PATH = ROOT_DIR / 'assets' / 'arial.ttf'

# --- Настройки ---
PDF_DPI = 200
CONFIDENCE = 0.2