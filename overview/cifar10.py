import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import yaml

from tqdm import tqdm

with open('./config/Alexnet_cifar10.yaml') as f:
    film = yaml.load(f, Loader=yaml.FullLoader)
    print(film)


parser = argparse.ArgumentParser(description='Train AlexNet on CIFAR-10')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes (default: 10)')
parser.add_argument('--print_period', type=int, default=100, help='how many batches to wait before logging training status (default: 100)')
parser.add_argument('--device', type=str, default='cuda', help='device to use for training (default: cuda)')
parser.add_argument('--data_root', type=str, default='./data', help='directory for storing input data (default: ./data)')
parser.add_argument('--save_root', type=str, default='./result/max_acc.pth', help='path to save the best model (default: ./result/max_acc.pth)')
parser.add_argument('--save_dir', type=str, default='./result', help='directory to save results (default: ./result)')

args = parser.parse_args()

breakpoint()



BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
EPOCH_NUM = args.epochs
NUM_CLASSES = args.num_classes
PRINT_PERIOD = args.print_period
DEVICE = args.device
DATA_ROOT = args.data_root
SAVE_ROOT = args.save_root
SAVE_DIR = args.save_dir



#epoch는 정의된 함수?


# Set device for training
device = torch.device(DEVICE)

# Define transformations for the input data
transform = transforms.Compose([
    transforms.Resize(224),  # Resize image to fit AlexNet input dimensions
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Load FashionMNIST dataset
trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True,  download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# Modify AlexNet for 1 channel input and 10 classes output
class AlexNet(nn.Module):
    def __init__(self, num_classes: int):

        super(AlexNet, self).__init__()
        self.features_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.GroupNorm(4, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.GroupNorm(4, 192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.GroupNorm(4, 384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.GroupNorm(4, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(4, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize the model
model = AlexNet(NUM_CLASSES).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)

# Function to train the model
def train_model():
    model.train()

    max_acc = -1
    for epoch in range(EPOCH_NUM):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, data in tqdm(enumerate(trainloader, 0)):
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % PRINT_PERIOD == 0:    # print every [PRINT_PERIOD] mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / PRINT_PERIOD:.3f}')
                running_loss = 0.0
              
        cur_acc = validate_model()
        if cur_acc >= max_acc:
            max_acc = cur_acc
            os.makedirs(SAVE_DIR,exist_ok=True)
            torch.save(model.state_dict(), SAVE_ROOT)
              
# Function to validate the model
def validate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')
    return accuracy

if __name__ == "__main__":
    train_model()
    validate_model()
