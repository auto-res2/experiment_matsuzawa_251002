"""src/preprocess.py
Dataset loading & augmentation for CIFAR-10 with support for
RandAugment magnitude coming from HyperParameters.  Falls back to the
previous *RandomTensorDataset* for smoke tests or when the user requests
`dataset_name: dummy`.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# ---------------------------------------------------------------------------
# Dummy dataset (kept for smoke tests) --------------------------------------
# ---------------------------------------------------------------------------

class RandomTensorDataset(Dataset):
    def __init__(self, length: int, input_shape: List[int], num_classes: int):
        super().__init__()
        g = torch.Generator().manual_seed(42)
        self.x = torch.randn(length, *input_shape, generator=g)
        self.y = torch.randint(0, num_classes, (length,), generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------------------
# CIFAR-10 helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

class CIFARDatasetWrapper(Dataset):
    """Wrap a HF *datasets* CIFAR entry to supply torch tensors."""

    def __init__(self, hf_dataset, transform):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["img"] if "img" in sample else sample["image"]
        label = sample["label"]
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def get_dataloaders(cfg, hp=None) -> Tuple[DataLoader, DataLoader]:
    """Return *train* and *validation* DataLoader according to *cfg*.

    Parameters
    ----------
    cfg : Dict
        Run-configuration dictionary.
    hp : HyperParameters | None
        If supplied, its *augment_magnitude* value is used inside the
        RandAugment transform for the training set.
    """
    dataset_name = cfg.get("dataset_name", "dummy")
    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 4))

    # ---------------------------------------------------------------------
    # Smoke-test dummy dataset -------------------------------------------
    # ---------------------------------------------------------------------
    if dataset_name == "dummy":
        input_shape = cfg.get("input_shape", [1, 28, 28])
        num_classes = int(cfg.get("num_classes", 10))
        length = int(cfg.get("dummy_length", 1024))
        train_ds = RandomTensorDataset(length, input_shape, num_classes)
        val_ds = RandomTensorDataset(length // 4, input_shape, num_classes)
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        )

    # ---------------------------------------------------------------------
    # CIFAR-10 real dataset via HuggingFace ------------------------------
    # ---------------------------------------------------------------------
    if dataset_name.lower() == "cifar10":
        ds = load_dataset("tanganke/cifar10")
        train_full = ds["train"]  # 50 000 examples

        # Stratified split 45 k / 5 k
        split = train_full.train_test_split(test_size=5000, seed=int(cfg.get("seed", 0)), stratify_by_column="label")
        train_ds_hf, val_ds_hf = split["train"], split["test"]

        # RandAugment magnitude – if HP passed use *current* value else cfg default
        ra_mag = 5
        if hp is not None:
            ra_mag = int(hp.augment_magnitude)
        else:
            ra_mag = int(cfg.get("hyperparams", {}).get("augment_magnitude", 5))

        train_tfms = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=ra_mag),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_tfms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_ds = CIFARDatasetWrapper(train_ds_hf, train_tfms)
        val_ds = CIFARDatasetWrapper(val_ds_hf, val_tfms)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader

    # ---------------------------------------------------------------------
    # Unknown dataset -----------------------------------------------------
    # ---------------------------------------------------------------------
    raise ValueError(f"Unknown dataset_name='{dataset_name}'. Supported: 'cifar10', 'dummy'.")
