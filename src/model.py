"""src/model.py
Model architecture registry.  Contains *baseline* implementation that works for
any generic classification smoke test.  Real experiments will replace or extend
these models.
"""

from __future__ import annotations

from typing import Dict

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


# ------------------------------- Registry ----------------------------------- #

_MODEL_REGISTRY: Dict[str, nn.Module] = {
    "BASE_CLASSIFIER": BaseClassifier,
    # ---------------------------------------------------------------------
    # PLACEHOLDER: Additional models will be registered here in later phase
    # ---------------------------------------------------------------------
}


def create_model(cfg: dict) -> nn.Module:
    mdl_cfg = cfg.get("model", {})
    name = mdl_cfg.get("name", "BASE_CLASSIFIER").upper()
    if name not in _MODEL_REGISTRY:
        raise NotImplementedError(
            f"Model '{name}' unknown to common core – must be provided in specialising step."
        )
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
