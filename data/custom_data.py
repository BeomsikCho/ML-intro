from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np

# class CustomDataset(Dataset):
#     def __init__(self,
#                  sample_num: int = 4000,
#                  sample_shape: list = [3, 224, 224],
#                  label_shape: list = [10]
#                  ):
#         """
#         sample_num (int): 데이터셋에 있는 sample 갯수
#         shape (int): 이미지 사이즈
#         """
#         super.__init__()
#         self.samples = []
#         self.labels = []
#         for _ in sample_num:
#             cur_sample = self.make_one_sample(sample_shape)
#             cur_label = self.make_one_label(label_shape)
#             self.samples.append(cur_sample)
#             self.labels.append(cur_label)
    
#     def make_one_sample(self, sample_shape):
#         return torch.randint(sample_shape)
    
#     def make_one_label(self, label_shape):
#         return torch.randint(label_shape)

#     def __getitem__(self, index):
#         cur_sample = self.samples[index]
#         label_sample = self.samples[index]
#         return cur_sample, label_sample
    
#     def __len__(self):
#         return len(self.samples)

from torchvision.datasets import ImageFolder


transform = transforms.Compose([        # Resize image to fit ResNet input dimensions
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   #   뒤쪽에서 계산하여 정의하기로 함
    ])

data_dir = ("./dataset/custom_dataset")
dataset = ImageFolder(root = data_dir, transform = transform)
testloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

# print("데이터 정보", dataset)
print(type(dataset[0]))
print(type(dataset[0][0]))
print(len(dataset))
print("데이터 모양",dataset[0][0].shape)
print("데이터 모양",dataset[1][0].shape)

meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in dataset]
stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in dataset]

meanR = np.mean([m[0] for m in meanRGB])
meanG = np.mean([m[1] for m in meanRGB])
meanB = np.mean([m[2] for m in meanRGB])

stdR = np.mean([s[0] for s in stdRGB])
stdG = np.mean([s[1] for s in stdRGB])
stdB = np.mean([s[2] for s in stdRGB])
print("평균",meanR, meanG, meanB)
print("표준편차",stdR, stdG, stdB)

# transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])  #  정규화(normalization)



# ImageFolder()

# CustomDataset[2]