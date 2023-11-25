#=============================================================================================#
                                    # перевод в 1 формат
from PIL import Image
import os
def convert_images_to_jpeg(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.tiff')):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            if image.format != 'png':
                new_image_path = os.path.splitext(image_path)[0] + ".png"
                image.save(new_image_path, "png")
                print(f"Converted {file_name} to PNG format.")
            else:
                print(f"{file_name} is already in PNG format.")
folder_path = "Папка с фото"
convert_images_to_jpeg(folder_path)
#=============================================================================================#