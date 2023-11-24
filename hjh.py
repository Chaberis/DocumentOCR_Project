import cv2
import numpy as np


def extract_text_words(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение порогового значения для получения черно-белого изображения
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Поиск контуров на изображении
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Итерация по контурам
    words = []
    for contour in contours:
        # Ограничивающий прямоугольник контура
        x, y, w, h = cv2.boundingRect(contour)

        # Исключение слишком маленьких и слишком больших контуров
        area = cv2.contourArea(contour)
        if area < 100 or area > 10000:
            continue

        # Вырезание слова из исходного изображения
        word_image = image[y:y + h, x:x + w]

        # Добавление слова в список
        words.append(word_image)

    return words


# Пример использования
image_path = '0.jpeg'  # Замените путь на путь к вашему изображению документа
word_images = extract_text_words(image_path)

# Сохранение каждого слова в отдельный файл в формате PNG
for i, word_image in enumerate(word_images):
    cv2.imwrite(f'word_{i}.png', word_image)  # Сохранение изображения с помощью OpenCV

# Вывод количества слов
print(f"Извлечено {len(word_images)} слов(a)")


