import torch.nn as nn
import torch.nn.functional as F

"""
Baseline model adapted from COMP4211 Spring 2021 PA2
"""


class PA2Net(nn.Module):
    def __init__(self, first_in_channel=1):
        super(PA2Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=first_in_channel, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.norm6 = nn.BatchNorm2d(512)
        self.pool2 = nn.AvgPool2d(16, 16, padding=0)

        self.fc1 = nn.Linear(512 * 1 * 1, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

        self.fully_connected = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        # Convolution layers for the images
        x = F.relu(self.norm1(self.conv1(x)))  # 32, 32, 32
        x = F.relu(self.norm2(self.conv2(x)))  # 32, 32, 32
        x = self.pool1(x)  # 32, 16, 16
        x = F.relu(self.norm3(self.conv3(x)))  # 64, 16, 16
        x = F.relu(self.norm4(self.conv4(x)))  # 128, 16, 16
        x = F.relu(self.norm5(self.conv5(x)))  # 256, 16, 16
        x = F.relu(self.norm6(self.conv6(x)))  # 512, 16, 16
        x = self.pool2(x)  # 512, 1, 1
        x = x.view(-1, 512)

        # Full connected layers and ouptut
        x = self.fully_connected(x)
        return x
