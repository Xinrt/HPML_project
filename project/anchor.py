import torch
from torchvision import transforms, datasets
from torchvision.models import resnet18, resnet34, resnet50, resnet101

# datapath
data_dir = '/scratch/xt2191/tiny-imagenet-200'

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load datasets
test_dataset = datasets.ImageFolder(root=data_dir + '/val', transform=data_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# load model
# model = resnet18(pretrained=False)
# model.load_state_dict(torch.hub.load_state_dict_from_url(
#     'https://download.pytorch.org/models/resnet18-5c106cde.pth'))

model = resnet34(pretrained=False)
model.load_state_dict(torch.hub.load_state_dict_from_url(
    'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth'))

# model = resnet50(pretrained=False)
# model.load_state_dict(torch.hub.load_state_dict_from_url(
#     'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth'))

# model = resnet101(pretrained=False)
# model.load_state_dict(torch.hub.load_state_dict_from_url(
#     'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth'))

model.eval()

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# test
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print("accuracy：{:.2f}".format(accuracy))
