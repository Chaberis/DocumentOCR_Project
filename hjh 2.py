import cv2
import numpy as np

def extract_text_words(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    words = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        area = cv2.contourArea(contour)
        if area < 100 or area > 10000:
            continue

        word_image = image[y:y + h, x:x + w]

        if w > 10 and h > 10:
            words.append(word_image)
    return words

image_path = '0.jpeg'
word_images = extract_text_words(image_path)
for i, word_image in enumerate(word_images):
    cv2.imwrite(f'word_{i}.png', word_image)
print("Извлечено", len(word_images), "слов(а)")
