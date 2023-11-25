import easyocr
import os

def recognize_handwritten_text(folder_path):
    reader = easyocr.Reader(['ru'], gpu=False)
    recognized_text_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            result = reader.readtext(image_path, detail=0)
            recognized_text_list.extend(result)

    return recognized_text_list

folder_path = 'dataset_test_1/handwritten'
recognized_handwritten_text = recognize_handwritten_text(folder_path)

print("Распознанный рукописный текст:")
print('\n'.join(recognized_handwritten_text))
