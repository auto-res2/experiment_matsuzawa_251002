"""src/model.py
Model factory with support for:
• `dummy`                 – tiny MLP for smoke tests
• `gpt2-small-scratch`    – GPT-2 small (~124 M) initialised from scratch
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn

# Optional heavy dep – only needed for GPT-2 runs
try:
    from transformers import GPT2Config, GPT2LMHeadModel  # type: ignore
except ImportError:  # pragma: no cover – transformers not required for smoke-tests
    GPT2Config = None  # type: ignore
    GPT2LMHeadModel = None  # type: ignore


class DummyModel(nn.Module):
    """Tiny two-layer MLP – used solely by smoke-tests."""

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

    def forward(self, x):  # noqa: D401
        return self.net(x)


# ---------------------------------------------------------------------------
# Factory -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_model(cfg):
    model_name = cfg.get("model_name", "dummy").lower()

    # ---------------- Dummy --------------------------------------------------
    if model_name == "dummy":
        input_shape = cfg.get("input_shape", [1, 28, 28])
        num_classes = int(cfg.get("num_classes", 10))
        return DummyModel(input_shape, num_classes)

    # ---------------- GPT-2 small from scratch ------------------------------
    if model_name == "gpt2-small-scratch":
        if GPT2Config is None or GPT2LMHeadModel is None:
            raise RuntimeError("transformers package required for GPT-2 model.")
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            resid_pdrop=float(cfg.get("residual_dropout", 0.1)),
            embd_pdrop=float(cfg.get("attention_dropout", 0.1)),
            attn_pdrop=float(cfg.get("attention_dropout", 0.1)),
        )
        model = GPT2LMHeadModel(config)
        return model

    # ---------------- Unknown ------------------------------------------------
    raise NotImplementedError(f"Unknown model_name='{model_name}'.")