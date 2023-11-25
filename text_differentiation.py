import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
#====================================================================================#

train_data_path = 'dataset'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = ImageFolder(train_data_path, transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
model = resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

device = torch.device("cpu")
model = model.to(device)

num_epochs = 6

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    print(f'Эпоха [{epoch + 1}/{num_epochs}], Потери: {total_loss:.4f}, Точность: {accuracy:.2f}%')

model_path = 'FragModel.pth'
#====================================================================================#

torch.save(model.state_dict(), model_path)
print(f'Модель сохранена по пути: {model_path}')