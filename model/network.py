import torch
from torch import nn as nn


class DMMRNet(nn.Module):
    # TODO: refactor this class to be more general, reusable and modular
    def __init__(self):
        super(DMMRNet, self).__init__()
        self.conv1 = nn.Conv3d(2, 32, kernel_size=3, padding=1, stride=4)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=4)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1, stride=4)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1, stride=4)
        self.bn4 = nn.BatchNorm3d(256)

        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, tar, src):
        x = torch.cat((tar, src), dim=1)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x.squeeze())


class DMMRTanh(DMMRNet):
    def __init__(self):
        super(DMMRTanh, self).__init__()

    def forward(self, tar, src):
        x = torch.cat((tar, src), dim=1)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.tanh(x.squeeze())
