from PIL import Image
import os
def convert_images_to_jpeg(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            if image.format != 'JPEG':
                new_image_path = os.path.splitext(image_path)[0] + ".jpeg"
                image.save(new_image_path, "JPEG")
                print(f"Converted {file_name} to JPEG format.")
            else:
                print(f"{file_name} is already in JPEG format.")
#Пример использования, для тупых максов
#folder_path = "/путь/к/папке"
#convert_images_to_jpeg(folder_path)
