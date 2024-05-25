import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import utils
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
import matplotlib.pyplot as plt

import numpy as np
import torch.optim as optim
import os
import argparse
import yaml
import time
import copy
from tqdm import tqdm

from lion_pytorch import Lion

# --------------------------------------- 건들지 말 것 ---------------------------------------
# Define argparse arguments
parser = argparse.ArgumentParser(description='Train RESNET on CIFAR-10')
parser.add_argument('--config_path', type=str, default='./config/RESNET_cifar10.yaml', help='Path to config YAML file')
args = parser.parse_args()

# Load configurations from YAML file
with open(args.config_path) as f:
    cfgs = yaml.load(f, Loader=yaml.FullLoader)


BATCH_SIZE = cfgs['data_cfg']['batch_size']
LEARNING_RATE = cfgs['train_cfg']['learning_rate']
EPOCH_NUM = cfgs['train_cfg']['epoch']
NUM_CLASSES = cfgs['model_cfg']['num_classes']
PRINT_PERIOD = cfgs['train_cfg']['print_period']
DEVICE = cfgs['setup_cfg']['device']
DATA_ROOT = cfgs['data_cfg']['data_root']
SAVE_ROOT = cfgs['train_cfg']['save_root']
SAVE_DIR = cfgs['train_cfg']['save_dir']


# Set device for training
device = torch.device(DEVICE)
# --------------------------------------- 건들지 말 것 ---------------------------------------


# Define transformations for the input data
transform = transforms.Compose([
    transforms.Resize(224),  # Resize image to fit AlexNet input dimensions
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Load FashionMNIST dataset
trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True,  download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)

testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

# residual block 정의
class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x
    
# ResNet 구조 정의
class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=10, init_weights=True):
        super().__init__()

        self.in_channels=64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(BottleNeck, [3,4,6,3])

def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])



# Initialize the model
model = resnet50().to(device)

x = torch.randn(3, 3, 224, 224).to(device)
output = model(x)
print(output.size())


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
optimizer = opt

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

# function to get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
    


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

        lr_scheduler.step(cur_acc)   

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

            # loss = criterion(outputs, labels) ??
            # running_loss += loss.item() ??

    
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')
    return accuracy

if __name__ == "__main__":
    train_model()
    validate_model()


# plot loss progress
plt.title("Train-Val Loss")
plt.plot(range(1,EPOCH_NUM+1),loss,label="train")
plt.plot(range(1,EPOCH_NUM+1),loss,label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()