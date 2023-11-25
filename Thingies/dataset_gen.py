from PIL import Image, ImageDraw, ImageFont
import random
import string
import os


def generate_dataset(num_samples, max_length):
    dataset = []
    output_folder = "images_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    possible_fonts = ["font1.ttf", "font2.ttf",
                      "font3.ttf", "font4.ttf"]

    for i in range(num_samples):
        length = random.randint(2, max_length)
        word = ''.join(random.choice('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬ \
                                     ЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789')
                       for _ in range(length))

        image_width = random.randint(80, 120)
        image_height = random.randint(40, 60)

        image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        font_file = random.choice(possible_fonts)
        font_size = random.randint(16, 20)
        font_path = os.path.join("", font_file)
        font = ImageFont.truetype(font_path, font_size)

        x = random.randint(image_width // 4, image_width // 2)
        y = random.randint(image_height // 4, image_height // 2)

        angle = random.randint(-10, 10)

        rotated_image = image.rotate(angle, expand=True)

        x_offset = random.randint(-5, 5)
        y_offset = random.randint(-5, 5)

        output_image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
        output_image.paste(rotated_image, (x_offset, y_offset))

        draw = ImageDraw.Draw(output_image)
        draw.text((x + x_offset, y + y_offset), word, fill=(0, 0, 0), font=font)

        output_image.save(os.path.join("dataset_final/printed/", f"printed{i}.png"))
        dataset.append(output_image)

    return dataset



num_samples = 40000
max_length = 6
dataset = generate_dataset(num_samples, max_length)
