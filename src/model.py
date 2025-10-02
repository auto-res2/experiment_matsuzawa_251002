"""src/model.py
Model-building utilities with clear placeholders for later injection of
real architectures. For smoke tests we supply a minimal *DummyModel* so
that the pipeline runs end-to-end without editing.
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn
import timm


class DummyModel(nn.Module):
    """A tiny MLP suitable only for smoke tests. Not task-specific."""

    def __init__(self, input_shape: List[int], num_classes: int):
        super().__init__()
        c, h, w = input_shape
        flat_dim = c * h * w
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):  # noqa: D401,E501  (simple forward pass)
        return self.net(x)


class BasicBlock(nn.Module):
    """Basic residual block for ResNet20/32/44/56/110 on CIFAR-10."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet20(nn.Module):
    """ResNet20 for CIFAR-10 (3 blocks of 3 layers each)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def build_model(cfg):
    """Return a torch.nn.Module according to `model_name` in cfg.

    Placeholders:
      • 'MODEL_PLACEHOLDER' or 'dummy' → returns DummyModel
      • otherwise: must be implemented later.
    """
    model_name = cfg.get("model_name", "MODEL_PLACEHOLDER")
    if model_name in {"dummy", "MODEL_PLACEHOLDER"}:
        input_shape = cfg.get("input_shape", [1, 28, 28])
        num_classes = int(cfg.get("num_classes", 10))
        return DummyModel(input_shape, num_classes)

    # ------------------------------------------------------------------
    # Real model architectures
    # ------------------------------------------------------------------
    if model_name == "resnet20":
        num_classes = int(cfg.get("num_classes", 10))
        return ResNet20(num_classes=num_classes)

    raise NotImplementedError(
        f"PLACEHOLDER: Unknown model_name='{model_name}'. Provide implementation in build_model().")
