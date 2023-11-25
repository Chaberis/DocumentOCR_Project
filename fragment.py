import cv2
import numpy as np


def extract_text_fragments(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Бинаризация изображения для получения черно-белого изображения
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Поиск контуров на изображении
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Итерации по контурам
    text_fragments = []
    for contour in contours:
        # Ограничивающий прямоугольник контура
        x, y, w, h = cv2.boundingRect(contour)

        # Вырезание фрагмента с тексом из исходного изображения
        text_fragment = image[y:y + h, x:x + w]

        # Добавление фрагмента с текстом в список
        text_fragments.append(text_fragment)

    return text_fragments


# Пример использования
image_path = 'Thingies/test_image.jpeg'  # Замените путь на путь к вашему изображению документа
text_fragments = extract_text_fragments(image_path)

# Сохранение фрагментов с текстом в отдельные файлы
for i, fragment in enumerate(text_fragments):
    cv2.imwrite(f'text_fragment_{i}.jpg', fragment)
