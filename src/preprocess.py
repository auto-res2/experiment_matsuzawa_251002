"""src/preprocess.py
Common data loading and preprocessing pipeline with *dataset placeholders*.
Only the generic mechanics are fully implemented here.  Any concrete dataset
integration must be supplied during the dataset-specific derivation step.

Current supported placeholder dataset:
    RANDOM_PLACEHOLDER – synthetic data used for smoke tests.
"""

from __future__ import annotations

import math
from typing import Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader

# -------------------------------- Placeholders ------------------------------ #

class RandomClassificationDataset(Dataset):
    """Synthetic dataset that produces random tensors for classification.

    This is *only* intended for smoke tests; real experiments must provide
    a concrete dataset loader by replacing the placeholder in config.
    """

    def __init__(self, num_samples: int, input_shape: Tuple[int, ...], num_classes: int):
        super().__init__()
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.input_shape)
        y = torch.randint(0, self.num_classes, (1,)).long().squeeze()
        return x, y


# -------------------------- Dataloader factory ------------------------------ #

def _create_placeholder_dataset(cfg: dict, split: str):
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"].upper()

    if name == "RANDOM_PLACEHOLDER":
        n_train = int(ds_cfg.get("num_samples", 1024))
        n_val = max(1, math.ceil(n_train * 0.2))
        if split == "train":
            dataset = RandomClassificationDataset(
                num_samples=n_train,
                input_shape=tuple(ds_cfg.get("input_shape", [1, 28, 28])),
                num_classes=int(ds_cfg.get("num_classes", 10)),
            )
        else:
            dataset = RandomClassificationDataset(
                num_samples=n_val,
                input_shape=tuple(ds_cfg.get("input_shape", [1, 28, 28])),
                num_classes=int(ds_cfg.get("num_classes", 10)),
            )
        return dataset

    # ---------------------------------------------------------------------
    # PLACEHOLDER: Will be replaced with specific dataset loading logic
    # ---------------------------------------------------------------------
    raise NotImplementedError(
        f"Dataset '{name}' not implemented in common core – must be provided in specialising step."
    )


def get_dataloader(cfg: dict, split: str = "train") -> DataLoader:
    """Return a PyTorch DataLoader for requested split.

    Parameters
    ----------
    cfg : dict
        Run-level configuration dictionary.
    split : str
        One of {"train", "val"}.
    """
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    ds = _create_placeholder_dataset(cfg, split)

    batch_size = int(cfg.get("training", {}).get("batch_size", 32))
    shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
