"""src/model.py
Model-building utilities with clear placeholders for later injection of
real architectures. For smoke tests we supply a minimal *DummyModel* so
that the pipeline runs end-to-end without editing.
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn


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
    # PLACEHOLDER: Real model architectures must be injected here.
    # ------------------------------------------------------------------
    raise NotImplementedError(
        f"PLACEHOLDER: Unknown model_name='{model_name}'. Provide implementation in build_model().")
