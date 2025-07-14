import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import streamlit as st
import cv2
import torch
from PIL import ImageFont
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pathlib import Path

import config
from processing import (
    load_image,
    count_pdf_pages,
    extract_text_data,
    create_final_image
)
from streamlit_image_comparison import image_comparison

st.set_page_config(layout="wide", page_title="DocuScribe")

@st.cache_resource
def load_models():
    """Загружает и кэширует модели ML."""
    models = {}
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models['detection'] = YOLO(config.DETECTION_MODEL)
        processor = TrOCRProcessor.from_pretrained(config.OCR_MODEL)
        ocr_model = VisionEncoderDecoderModel.from_pretrained(config.OCR_MODEL).to(device) # type: ignore
        models['ocr'] = {'processor': processor, 'model': ocr_model, 'device': device}
        
        try:
            ImageFont.truetype(config.FONT_PATH, 10)
            models['font_path'] = config.FONT_PATH
        except IOError:
            models['font_path'] = None
            st.warning(f"Шрифт не найден: {config.FONT_PATH}")
            
    except Exception as e:
        st.error(f"Ошибка загрузки моделей: {e}")
        return None
    return models

def init_session_state():
    """Инициализирует состояние сессии, если оно еще не создано."""
    if "file_id" not in st.session_state:
        st.session_state.file_id = None
        st.session_state.source_image = None
        st.session_state.processed_image = None
        st.session_state.text_data = []
        st.session_state.file_name = None

def main():
    """Основная функция UI приложения."""
    st.title("📝 Docu`Scribe`")
    st.markdown("Замените рукописный текст на печатный.")
    
    init_session_state()
    models = load_models()
    if not models:
        return

    with st.sidebar:
        st.header("Настройки")
        conf = st.slider("Порог уверенности", 0.0, 1.0, config.CONFIDENCE, 0.05)
    
    uploaded_file = st.file_uploader("Загрузите изображение или PDF", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file and uploaded_file.file_id != st.session_state.file_id:
        st.session_state.file_id = uploaded_file.file_id
        st.session_state.file_name = uploaded_file.name
        
        page_num = 0
        if uploaded_file.type == "application/pdf":
            num_pages = count_pdf_pages(uploaded_file.getvalue())
            if num_pages > 1:
                page_num = st.slider("Выберите страницу", 1, num_pages, 1) - 1
        
        try:
            image = load_image(uploaded_file, page_num=page_num)
            if image is not None:
                st.session_state.source_image = image.copy()
                st.session_state.text_data = []
                st.session_state.processed_image = None
                st.rerun()
        except IOError as e:
            st.error(str(e)) 
            st.session_state.source_image = None

    if st.session_state.source_image is not None:
        if st.button("🚀 Начать обработку", use_container_width=True):
            with st.spinner('Обработка...'):
                text_data = extract_text_data(st.session_state.source_image, models, conf)
                st.session_state.text_data = text_data
                if text_data:
                    processed_image = create_final_image(st.session_state.source_image, text_data, models['font_path'])
                    st.session_state.processed_image = processed_image
                else:
                    st.warning("Текст не найден.")
            st.rerun()

    if st.session_state.processed_image is not None:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Результат")
            with st.container(height=800):
                original_rgb = cv2.cvtColor(st.session_state.source_image, cv2.COLOR_BGR2RGB) # type: ignore
                processed_rgb = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                image_comparison(img1=original_rgb, img2=processed_rgb, label1="Оригинал", label2="Результат")

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                _, buf = cv2.imencode(".png", st.session_state.processed_image)
                st.download_button("Скачать .png", buf.tobytes(), f"{Path(st.session_state.file_name).stem}_processed.png", "image/png", use_container_width=True)
            with dl_col2:
                full_text = "\n".join([item['text'] for item in st.session_state.text_data if item['text']])
                st.download_button("Скачать .txt", full_text.encode('utf-8'), f"{Path(st.session_state.file_name).stem}_text.txt", "text/plain", use_container_width=True)

        with col2:
            st.subheader("Редактор текста")
            if st.session_state.text_data:
                with st.container(height=800):
                    edited_texts = {}
                    for i, item in enumerate(st.session_state.text_data):
                        edited_texts[i] = st.text_input(f"Фрагмент {i+1}", value=item['text'], key=f"text_{i}")

                if st.button("✅ Применить", use_container_width=True):
                    for i, new_text in edited_texts.items():
                        st.session_state.text_data[i]['text'] = new_text
                    
                    new_image = create_final_image(st.session_state.source_image, st.session_state.text_data, models['font_path'])
                    st.session_state.processed_image = new_image
                    st.rerun()
            else:
                st.info("Нет текста для редактирования.")

if __name__ == "__main__":
    main()