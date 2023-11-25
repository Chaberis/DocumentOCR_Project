from PIL import Image
import os


def convert_images_to_png(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.tiff')):
            image_path = os.path.join(folder_path, file_name)
            end_path = os.path.join("place_docs_here/processing/pic_to_png", file_name)
            image = Image.open(image_path)
            if image.format != 'png':
                new_image_path = os.path.splitext(end_path)[0] + ".png"
                image.save(new_image_path, "png")


folder_path = "place_docs_here"
convert_images_to_png(folder_path)
