import torch
import torch.nn as nn




# 입력 데이터의 크기: (배치 크기, 채널, 높이, 너비)
input_size = (64, 3, 32, 32)

# Conv2d 레이어 정의
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

# 입력 데이터 생성
input_data = torch.randn(input_size)

# 컨볼루션 연산 수행
output = conv(input_data)

# 출력 데이터의 크기 출력
print("Output size:", output.size())

Output size: torch.Size([64, 64, 32, 32])