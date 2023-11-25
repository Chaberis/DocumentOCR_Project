import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
from torch import nn, optim
import os

model_path = 'FragModel.pth'
model = resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()


def classify_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()
    return predicted_class


def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = os.listdir(input_folder)

    for image_file in images:
        image_path = os.path.join(input_folder, image_file)
        predicted_class = classify_image(image_path)

        if predicted_class == 1:
            save_path = os.path.join(output_folder, image_file)
            os.rename(image_path, save_path)


input_folder = 'place_docs_here/processing/segmented_pngs'
output_folder = 'place_docs_here/processing/differented_pngs'

process_images_in_folder(input_folder, output_folder)
