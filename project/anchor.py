import torch
from torchvision import transforms, datasets
from torchvision.models import resnet18

# 数据集路径
data_dir = '/scratch/xt2191/tiny-imagenet-200'

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载测试集
test_dataset = datasets.ImageFolder(root=data_dir + '/val', transform=data_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载ResNet-18模型
model = resnet18(pretrained=True)
model.eval()

# GPU设备（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 测试模型准确度
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
print("准确度：{:.2f}%".format(accuracy))