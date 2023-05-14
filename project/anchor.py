import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from timm import create_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
resnet34_url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth'
resnet50_url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth'
resnet101_url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth'

# load model
# model = create_model('resnet18', pretrained=False)
# model.load_state_dict(torch.hub.load_state_dict_from_url(resnet18_url))

# model = create_model('resnet34', pretrained=False)
# model.load_state_dict(torch.hub.load_state_dict_from_url(resnet34_url))

# model = create_model('resnet50', pretrained=False)
# model.load_state_dict(torch.hub.load_state_dict_from_url(resnet50_url))

model = create_model('resnet101', pretrained=False)
model.load_state_dict(torch.hub.load_state_dict_from_url(resnet101_url))




model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = '/scratch/xt2191/tiny-imagenet-200'
dataset = ImageFolder(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model.eval()

# test
correct = 0
total = 0
with torch.no_grad():
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
