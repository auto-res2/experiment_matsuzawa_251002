"""src/preprocess.py
Dataset loading and preprocessing pipeline for CIFAR-10 (full experiments) and
for a small synthetic dataset (smoke tests).
"""

from __future__ import annotations

import math
from typing import Tuple, Any, Dict

import torch
from torch.utils.data import Dataset, DataLoader, Subset

# -------------------------------- Synthetic data --------------------------- #

class RandomClassificationDataset(Dataset):
    """Synthetic dataset that produces random tensors for classification.
    This is only used by the smoke tests and keeps runtime negligible."""

    def __init__(self, num_samples: int, input_shape: Tuple[int, ...], num_classes: int):
        super().__init__()
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes

    def __len__(self) -> int:  # type: ignore[override]
        return self.num_samples

    def __getitem__(self, idx: int):  # type: ignore[override]
        x = torch.randn(self.input_shape)
        y = torch.randint(0, self.num_classes, (1,)).long().squeeze()
        return x, y


# ----------------------------- CIFAR-10 helpers ----------------------------- #

from torchvision import datasets, transforms  # noqa: E402

class Cutout(object):
    """Simple Cutout augmentation (applied after normalization)."""

    def __init__(self, length: int = 16):
        self.length = length

    def __call__(self, img: torch.Tensor):  # expects tensor of shape C×H×W
        h, w = img.shape[1], img.shape[2]
        cy = torch.randint(0, h, (1,)).item()
        cx = torch.randint(0, w, (1,)).item()
        y1 = max(0, cy - self.length // 2)
        y2 = min(h, cy + self.length // 2)
        x1 = max(0, cx - self.length // 2)
        x2 = min(w, cx + self.length // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img


class _SubsetWithTransform(Dataset):
    """A thin wrapper that allows different transforms for subset splits."""

    def __init__(self, subset: Subset, transform):
        super().__init__()
        self.subset = subset
        self.transform = transform

    def __len__(self):  # type: ignore[override]
        return len(self.subset)

    def __getitem__(self, idx):  # type: ignore[override]
        img, label = self.subset[idx]
        img = self.transform(img)
        return img, label


_CIFAR10_CACHE: Dict[Tuple[str, int], Dict[str, Dataset]] = {}


def _get_cifar10_dataset(cfg: Dict[str, Any], split: str) -> Dataset:
    ds_cfg = cfg["dataset"]
    seed = int(cfg.get("seed", 42))
    root = ds_cfg.get("root", "./data")
    val_split = int(ds_cfg.get("val_split", 5000))

    cache_key = (root, seed)
    if cache_key not in _CIFAR10_CACHE:
        full_train = datasets.CIFAR10(root=root, train=True, download=True)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True)

        num_train = len(full_train)  # 50000
        indices = torch.randperm(num_train, generator=torch.Generator().manual_seed(seed))
        train_idx = indices[val_split:]
        val_idx = indices[:val_split]
        train_subset = Subset(full_train, train_idx)
        val_subset = Subset(full_train, val_idx)
        _CIFAR10_CACHE[cache_key] = {
            "train": train_subset,
            "val": val_subset,
            "test": test_ds,
        }

    base_ds = _CIFAR10_CACHE[cache_key][split]

    # Build transforms
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    aug_cfg = ds_cfg.get("augment", {})
    magnitude = int(aug_cfg.get("magnitude", 9))
    cutout_len = int(aug_cfg.get("cutout_len", 16))

    if split == "train":
        tr_list = []
        if magnitude > 0:
            tr_list.append(transforms.RandAugment(num_ops=2, magnitude=magnitude))
        tr_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout(cutout_len),
        ])
    else:
        tr_list = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    transform = transforms.Compose(tr_list)
    return _SubsetWithTransform(base_ds, transform)


# -------------------------- Dataset factory --------------------------------- #

def _create_dataset(cfg: dict, split: str) -> Dataset:
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"].upper()

    if name == "SYNTHETIC":
        n_train = int(ds_cfg.get("num_samples", 1024))
        n_val = max(1, math.ceil(n_train * 0.2))
        if split == "train":
            return RandomClassificationDataset(
                num_samples=n_train,
                input_shape=tuple(ds_cfg.get("input_shape", [1, 28, 28])),
                num_classes=int(ds_cfg.get("num_classes", 10)),
            )
        else:
            return RandomClassificationDataset(
                num_samples=n_val,
                input_shape=tuple(ds_cfg.get("input_shape", [1, 28, 28])),
                num_classes=int(ds_cfg.get("num_classes", 10)),
            )

    if name == "CIFAR10":
        return _get_cifar10_dataset(cfg, split)

    raise NotImplementedError(f"Dataset '{name}' is not supported.")


# ----------------------------- Public API ----------------------------------- #

def get_dataloader(cfg: dict, split: str = "train") -> DataLoader:
    """Return a PyTorch DataLoader for the requested split."""
    assert split in {"train", "val", "test"}, "split must be 'train', 'val', or 'test'"
    ds = _create_dataset(cfg, split)
    batch_size = int(cfg.get("training", {}).get("batch_size", 32))
    shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
