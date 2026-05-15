import torch
import torch.nn as nn

class LeNet(torch.nn.Module):

    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=6, kernel_size=3, padding='same', bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=6)
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
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def _threshold_init(self, layer):
        if isinstance(layer, nn.Conv2d):
            layer.hard_threshold = None


    def forward(self, x):
        x = self.maxpool(self.bn1(self.relu(self.conv1(x)))) # 
        x = self.maxpool(self.bn2(self.relu(self.conv2(x))))
        x = self.flatten(x)
        x = self.bn3(self.relu(self.lin1(x)))
        x = self.bn4(self.relu(self.lin2(x)))
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
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
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


class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet20(nn.Module):
    """ResNet-20 architecture for CIFAR-10 (32x32 RGB images)"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 16
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks: 3 groups of 3 blocks each
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)
        
        # Global average pooling and classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes, bias=False)
        
        # Weight initialization
        self.apply(self._weight_init)
        self.apply(self._threshold_init)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """Create a residual layer with multiple blocks"""
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _weight_init(self, layer):
        """Kaiming initialization for conv and linear layers"""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    def _threshold_init(self, layer):
        """Initialize threshold for conv layers"""
        if isinstance(layer, nn.Conv2d):
            layer.hard_threshold = None
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Classification head
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x


