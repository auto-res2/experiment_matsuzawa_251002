"""Model architectures & registry."""

from typing import Dict, Any
import torch
import torch.nn as nn

__all__ = [
    "get_model",
]

###############################################################################
#                           Vision toy models                                 #
###############################################################################

class MLP(nn.Module):
    """Simple MLP for flattened inputs."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dims=(256, 128)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(inplace=True)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        return self.net(x)


class SimpleCNN(nn.Module):
    """Minimal CNN for 32×32 images."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

###############################################################################
#                          GPT-2 small language model                         #
###############################################################################

class GPT2SmallLM(nn.Module):
    """GPT-2 small (124 M params) instantiated from scratch for language modelling."""

    def __init__(self, vocab_size: int = 50257, max_seq_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError as e:
            raise ImportError("transformers package is required for GPT2SmallLM") from e

        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_seq_len,
            n_ctx=max_seq_len,
            n_embd=768,
            n_layer=12,
            n_head=12,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.model = GPT2LMHeadModel(config)

    def forward(self, input_ids, labels=None):  # type: ignore[override]
        return self.model(input_ids=input_ids, labels=labels)

###############################################################################
#                              Registry                                       #
###############################################################################

_MODEL_REGISTRY = {
    "MLP": MLP,
    "SimpleCNN": SimpleCNN,
    "GPT2SmallLM": GPT2SmallLM,
}

###############################################################################
#                             Factory                                         #
###############################################################################

def get_model(model_cfg: Dict[str, Any], *, num_classes: int, input_shape: Any) -> nn.Module:  # noqa: ANN401
    model_type = model_cfg["type"]
    if model_type == "MLP":
        flat_dim = int(torch.prod(torch.tensor(input_shape))) if isinstance(input_shape, (tuple, list)) else int(input_shape)
        return MLP(flat_dim, num_classes)
    elif model_type == "SimpleCNN":
        return SimpleCNN(num_classes)
    elif model_type == "GPT2SmallLM":
        vocab_size = model_cfg.get("vocab_size", 50257)
        max_seq_len = model_cfg.get("max_seq_len", input_shape[0] if input_shape is not None else 1024)
        return GPT2SmallLM(vocab_size=vocab_size, max_seq_len=max_seq_len)
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Available: {list(_MODEL_REGISTRY.keys())}")
