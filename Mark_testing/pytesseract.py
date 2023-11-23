import pytesseract
from PIL import Image

def replace_text_with_white_background(image_path):

    image = Image.open(image_path).convert("RGB")

    text = pytesseract.image_to_string(image)

    print("Распознанный текст:")
    print(text)

    boxes = pytesseract.image_to_boxes(image)

    width, height = image.size
    bboxes = []
    for box in boxes.splitlines():
        _, x, y, w, h = box.split(" ")
        bboxes.append((int(x), height - int(y), int(w), int(h)))

    for bbox in bboxes:
        x, y, w, h = bbox
        image.paste((255, 255, 255), box=(x, y, x + w, y + h))

    new_image_path = image_path.split(".")[0] + "_modified.jpeg"
    image.save(new_image_path)

    print("Текст успешно заменен.")

image_path = "путь/к/изображению.jpeg"
replace_text_with_white_background(image_path)
