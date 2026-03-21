import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.stride = stride
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet_Layerwise(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet_Layerwise, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        """
        前向传播，支持返回中间特征
        
        Args:
            x: 输入张量
            return_features: 是否返回中间特征
            
        Returns:
            如果return_features为False，返回最终输出
            如果return_features为True，返回字典包含各层特征
        """
        features = {}
        
        out = self.relu(self.conv1(x))
        features['conv1'] = out
        
        out = self.layer1(out)
        features['layer1'] = out
        
        out = self.layer2(out)
        features['layer2'] = out
        
        out = self.layer3(out)
        features['layer3'] = out
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        features['avgpool'] = out
        
        out = self.fc(out)
        features['final'] = out
        
        if return_features:
            return features
        else:
            return out

def resnet20_cifar():
    """
    创建ResNet-20模型
    """
    return ResNet_Layerwise(BasicBlock, [3, 3, 3])

def resnet32_cifar():
    """
    创建ResNet-32模型
    """
    return ResNet_Layerwise(BasicBlock, [5, 5, 5])

def resnet44_cifar():
    """
    创建ResNet-44模型
    """
    return ResNet_Layerwise(BasicBlock, [7, 7, 7])

def resnet56_cifar():
    """
    创建ResNet-56模型
    """
    return ResNet_Layerwise(BasicBlock, [9, 9, 9])