import os
from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from huggingface_hub import hf_hub_download
import streamlit as st
import config

def _download_ocr_model():
    """Скачивает модель OCR с Hugging Face Hub, если она отсутствует."""
    if config.OCR_MODEL_PATH.exists() and any(config.OCR_MODEL_PATH.iterdir()):
        return
    
    st.info(f"Скачивание модели распознавания текста: {config.OCR_REPO_ID}...")
    try:
        processor = TrOCRProcessor.from_pretrained(config.OCR_REPO_ID)
        model = VisionEncoderDecoderModel.from_pretrained(config.OCR_REPO_ID)
        
        config.OCR_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        
        processor.save_pretrained(config.OCR_MODEL_PATH) # type: ignore
        model.save_pretrained(config.OCR_MODEL_PATH)
        st.info(f"Модель распознавания текста сохранена.")
    except Exception as e:
        st.error(f"Ошибка скачивания модели OCR: {e}")
        raise

def _download_detection_model():
    """Скачивает модель детекции с Hugging Face Hub, если она отсутствует."""
    if config.DETECTION_MODEL_PATH.exists():
        return

    st.info(f"Скачивание модели детекции: {config.DETECTION_REPO_ID}...")
    try:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=config.DETECTION_REPO_ID,
            filename=config.DETECTION_FILENAME,
            local_dir=config.MODELS_DIR,
            local_dir_use_symlinks=False
        )
        st.info(f"Модель детекции сохранена.")
    except Exception as e:
        st.error(f"Ошибка скачивания модели детекции: {e}")
        raise

@st.cache_resource
def download_models():
    """
    Проверяет наличие всех необходимых моделей и скачивает их при отсутствии.
    """
    try:
        _download_ocr_model()
        _download_detection_model()
        return True
    except Exception:
        return False

if __name__ == "__main__":
    print("--- Проверка и скачивание моделей ---")
    # Для локального запуска без Streamlit, st.* работать не будет.
    # Этот блок теперь больше для информации.
    if config.OCR_MODEL_PATH.exists() and config.DETECTION_MODEL_PATH.exists():
        print("Модели уже существуют локально.")
    else:
        print("Запустите приложение Streamlit для автоматического скачивания моделей.")