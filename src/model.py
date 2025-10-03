"""src/model.py
Model architecture registry for experiments:
  • BASE_CLASSIFIER – tiny CNN used only for smoke tests
  • RESNET20        – classic ResNet-20 for CIFAR-10
  • RESNET50        – torchvision ResNet-50 (final fc adapted to num_classes)
"""

from __future__ import annotations

from typing import Dict, Callable

import torch
import torch.nn as nn

# ----------------------------- Base classifier ------------------------------ #

class BaseClassifier(nn.Module):
    """Very small CNN suitable for 1×28×28 images (MNIST-like)."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------------- ResNet-20 (CIFAR) ---------------------------- #

class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut: nn.Module
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, block: Callable, num_blocks: list[int], num_classes: int = 10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Callable, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet20(num_classes: int = 10):
    return ResNetCIFAR(_BasicBlock, [3, 3, 3], num_classes=num_classes)


# ----------------------------- ResNet-50 (torchvision) ----------------------- #

from torchvision import models  # noqa: E402

def ResNet50(num_classes: int = 10):
    model = models.resnet50(weights=None)  # train from scratch as per experiment
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ------------------------------- Registry ----------------------------------- #

_MODEL_REGISTRY: Dict[str, Callable] = {
    "BASE_CLASSIFIER": BaseClassifier,
    "RESNET20": ResNet20,
    "RESNET50": ResNet50,
}


def create_model(cfg: dict) -> nn.Module:
    mdl_cfg = cfg.get("model", {})
    name = mdl_cfg.get("name", "BASE_CLASSIFIER").upper()
    if name not in _MODEL_REGISTRY:
        raise NotImplementedError(f"Model '{name}' is not registered.")
    kwargs = mdl_cfg.get("kwargs", {})
    return _MODEL_REGISTRY[name](**kwargs)


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(model_class_name: str, path: str, device: torch.device) -> nn.Module:
    model_class_name = model_class_name.upper()
    if model_class_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model class '{model_class_name}' for loading")
    model = _MODEL_REGISTRY[model_class_name]()
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
