import os
from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import config


OCR_HUB_ID = 'kazars24/trocr-base-handwritten-ru'

def setup_ocr_model():
    """Скачивает модель OCR, если она отсутствует."""
    if config.OCR_MODEL.exists() and any(config.OCR_MODEL.iterdir()):
        print(f"Модель OCR найдена: {config.OCR_MODEL}")
        return True

    print(f"Скачивание модели: {OCR_HUB_ID}...")
    try:
        processor = TrOCRProcessor.from_pretrained(OCR_HUB_ID)
        model = VisionEncoderDecoderModel.from_pretrained(OCR_HUB_ID)
        
        config.OCR_MODEL.mkdir(parents=True, exist_ok=True)
        
        processor.save_pretrained(config.OCR_MODEL) # type: ignore
        model.save_pretrained(config.OCR_MODEL)
        
        print(f"Модель сохранена в: {config.OCR_MODEL}")
        return True
    except Exception as e:
        print(f"Ошибка скачивания модели OCR: {e}")
        return False

def check_detection_model():
    """Проверяет наличие модели детекции."""
    if config.DETECTION_MODEL.exists():
        print(f"Модель детекции найдена: {config.DETECTION_MODEL}")
        return True
    
    print(f"ОШИБКА: Модель детекции не найдена по пути {config.DETECTION_MODEL}")
    print("Пожалуйста, скачайте файл 'best.pt' и поместите его в папку 'models'.")
    return False

if __name__ == "__main__":
    print("--- Проверка моделей ---")
    ocr_ok = setup_ocr_model()
    detection_ok = check_detection_model()
    
    if ocr_ok and detection_ok:
        print("\nВсе модели готовы.")
    else:
        print("\nНе все модели готовы. Запустите приложение после устранения ошибок.")