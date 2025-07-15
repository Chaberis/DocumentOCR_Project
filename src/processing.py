import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import config

def count_pdf_pages(file_bytes):
    """Возвращает количество страниц в PDF."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return doc.page_count
    except Exception:
        return 0

def load_image(uploaded_file, page_num=0):
    """Конвертирует загруженный файл (изображение или PDF) в изображение OpenCV."""
    file_bytes = uploaded_file.read()
    
    if uploaded_file.type == "application/pdf":
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=config.PDF_DPI) # type: ignore
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            uploaded_file.seek(0)
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) if img.shape[2] == 4 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise IOError(f"Не удалось обработать PDF: {e}") from e
    else:
        img_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def run_ocr(image, ocr):
    """Распознает текст на фрагменте изображения."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = ocr['processor'](images=img_rgb, return_tensors="pt").pixel_values.to(ocr['device'])
    generated_ids = ocr['model'].generate(pixels)
    return ocr['processor'].batch_decode(generated_ids, skip_special_tokens=True)[0]

def draw_text_on_image(image, box, text, font_path):
    """Рисует полупрозрачную плашку и накладывает на нее текст."""
    x1, y1, x2, y2 = [max(0, val) for val in box]
    if y2 <= y1 or x2 <= x1: 
        return image

    sub_img = image[y1:y2, x1:x2]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    res = cv2.addWeighted(sub_img, 0.2, white_rect, 0.8, 1.0)
    image[y1:y2, x1:x2] = res

    if font_path:
        try:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            font_size = max(10, int((y2 - y1) * 0.7))
            font = ImageFont.truetype(font_path, font_size)
            text_y = y1 + ((y2 - y1 - font_size) // 2)
            draw.text((x1 + 5, text_y), text, font=font, fill=(0, 0, 0))
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error rendering text: {e}")
            return image
    return image

def extract_text_data(image, models, conf):
    """Выполняет детекцию и OCR, возвращая список словарей с текстом и координатами."""
    print("--- Начало extract_text_data ---")
    results = models['detection'](image, conf=conf)
    boxes = [[int(i) for i in box.xyxy[0]] for result in results for box in result.boxes]
    
    print(f"Найдено {len(boxes)} фрагментов для распознавания.")
    
    if not boxes:
        print("--- Завершение extract_text_data (фрагменты не найдены) ---")
        return []

    text_data = []
    for i, box in enumerate(boxes):
        print(f"Обработка фрагмента {i+1}/{len(boxes)}...")
        roi = image[box[1]:box[3], box[0]:box[2]]
        if roi.size > 0:
            text = run_ocr(roi, models['ocr'])
            text_data.append({'box': box, 'text': text})
            print(f"  > Фрагмент {i+1} распознан. Результат: '{text[:30]}...'")
        else:
            print(f"  > Фрагмент {i+1} пропущен (нулевой размер).")
            
    print("--- Завершение extract_text_data (успешно) ---")
    return text_data

def create_final_image(image, text_data, font_path):
    """Наносит данные на исходное изображение."""
    output_image = image.copy()
    for item in text_data:
        if item['text']:
            output_image = draw_text_on_image(output_image, item['box'], item['text'], font_path)
    return output_image