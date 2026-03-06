import torch
import torchvision
import torch.nn as nn
from irwl1.config import BATCH_SIZE

class LeNet(torch.nn.Module):

    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=6, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(in_features=576, out_features=120, bias=False)
        self.bn3 = nn.BatchNorm1d(num_features=120)

        self.lin2 = nn.Linear(in_features=120, out_features=84, bias=False)
        self.bn4 = nn.BatchNorm1d(num_features=84)
        self.out = nn.Linear(in_features=84, out_features=10)

        self.apply(self._weight_init)
        self.apply(self._threshold_init)

    def _weight_init(self, layer):
        if type(layer) in [nn.Linear, nn.Conv2d]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def _threshold_init(self, layer):
        if type(layer) in [nn.Conv2d]:
            layer.hard_threshold = None


    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.relu(self.bn3(self.lin1(x)))
        x = self.relu(self.bn4(self.lin2(x)))
        y = self.out(x)
        return y


# original LeNet without batch normalization
class oriLeNet(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=6, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)

        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(in_features=576, out_features=120)

        self.lin2 = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=10)

        self.apply(self._weight_init)
        self.apply(self._threshold_init)

    def _weight_init(self, layer):
        if type(layer) in [nn.Linear, nn.Conv2d]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def _threshold_init(self, layer):
        if type(layer) in [nn.Conv2d]:
            layer.hard_threshold = None


    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        y = self.out(x)
        return y


