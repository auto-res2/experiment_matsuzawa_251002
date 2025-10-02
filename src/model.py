"""Model definitions & registry for CIFAR-10 experiments (ResNet-20 / ResNet-50).
The implementations follow the originalTiny ResNet paper (He et al. 2016) with
minor adaptations for 32×32 images."""

from typing import Dict, Any

import torch
import torch.nn as nn


#########################################
#               Blocks                 #
#########################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


#########################################
#              ResNet                 #
#########################################

class _ResNetCIFAR(nn.Module):
    """Generic ResNet for CIFAR-style small images."""

    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # Parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet20(_ResNetCIFAR):
    def __init__(self, num_classes=10):
        super().__init__(BasicBlock, [3, 3, 3], num_classes=num_classes)


class ResNet32(_ResNetCIFAR):
    def __init__(self, num_classes=10):
        super().__init__(BasicBlock, [5, 5, 5], num_classes=num_classes)


class ResNet44(_ResNetCIFAR):
    def __init__(self, num_classes=10):
        super().__init__(BasicBlock, [7, 7, 7], num_classes=num_classes)


class ResNet56(_ResNetCIFAR):
    def __init__(self, num_classes=10):
        super().__init__(BasicBlock, [9, 9, 9], num_classes=num_classes)


class ResNet50(_ResNetCIFAR):
    """Bottleneck-based ResNet-50 adapted for 32×32 images (depth 50)."""

    def __init__(self, num_classes=10):
        super().__init__(Bottleneck, [3, 4, 6], num_classes=num_classes)


#########################################
#              Registry               #
#########################################

_MODEL_REGISTRY = {
    "ResNet20": ResNet20,
    "ResNet32": ResNet32,
    "ResNet44": ResNet44,
    "ResNet56": ResNet56,
    "ResNet50": ResNet50,
}


def get_model(model_cfg: Dict[str, Any], *, num_classes: int, input_shape):
    model_type = model_cfg["type"]
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[model_type](num_classes=num_classes)
