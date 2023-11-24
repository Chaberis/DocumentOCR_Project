import easyocr


def recognize_handwritten_text(image_path):
    reader_hwr = easyocr.Reader(['ru'], gpu=False)
    reader_ocr = easyocr.Reader(['ru'], gpu=False)
    result_hwr = reader_hwr.readtext(image_path, detail=0)
    result_ocr = reader_ocr.readtext(image_path, detail=0)

    handwritten_text = []

    for text in result_hwr:
        if text in result_ocr:
            handwritten_text.append(text)

    return handwritten_text

image_path = 'dataset_test_1/handwritten/ad3742.png'
recognized_handwritten_text = recognize_handwritten_text(image_path)
print("Распознанный рукописный текст:")
print('\n'.join(recognized_handwritten_text))
