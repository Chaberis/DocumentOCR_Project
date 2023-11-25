#====================================================================================#
                                #проверка класса
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
    output_folder = 'classified_images'
    os.makedirs(output_folder, exist_ok=True)

    if predicted_class == 1:
        image_name = os.path.basename(image_path)
        save_path = os.path.join(output_folder, image_name)
        os.rename(image_path, save_path)
    return predicted_class

image_path = 'dataset_test_1/handwritten/ab3742.png'
predicted_class = classify_image(image_path)
print(f'Предсказанный класс: {predicted_class}')
#====================================================================================#