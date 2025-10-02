"""Data loading & augmentation utilities for vision benchmarks (CIFAR-10 for this
experiment) and a FakeData fallback for CI smoke tests."""

from typing import Tuple, Any, Dict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def _cifar10_transforms(train: bool, randaugment_magnitude: int, cutout: bool):
    """Return torchvision transforms suitable for CIFAR-10 training / evaluation."""
    if train:
        augments = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if randaugment_magnitude > 0:
            augments.append(transforms.RandAugment(num_ops=2, magnitude=randaugment_magnitude))
        augments.append(transforms.ToTensor())
        augments.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)))
        if cutout:
            # Use RandomErasing as Cutout analogue
            augments.append(transforms.RandomErasing(p=0.5, scale=(0.05, 0.15)))
        return transforms.Compose(augments)
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ]
        )


def get_dataloaders(
    dataset_cfg: Dict[str, Any],
    *,
    batch_size: int,
    num_workers: int,
    smoke_test: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, Any]:
    """Create train / val / test dataloaders.

    Args:
        dataset_cfg: configuration dict
        batch_size: per-loader batch size
        num_workers: dataloader workers
        smoke_test: drastically reduce dataset sizes for CI

    Returns:
        train_loader, val_loader, test_loader, num_classes, input_shape
    """

    dataset_type = dataset_cfg["type"].lower()
    data_root = Path(dataset_cfg.get("root", "./data"))
    rand_m = int(dataset_cfg.get("randaugment_magnitude", 9))
    cutout = bool(dataset_cfg.get("cutout", True))
    val_split = float(dataset_cfg.get("val_split", 0.1))

    if dataset_type == "cifar10":
        train_transform = _cifar10_transforms(True, rand_m, cutout)
        test_transform = _cifar10_transforms(False, rand_m, cutout)

        full_train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)

        val_size = int(len(full_train) * val_split)
        train_size = len(full_train) - val_size
        train_set, val_set = random_split(full_train, [train_size, val_size])

        num_classes = 10
        input_shape = (3, 32, 32)

    elif dataset_type == "fakedata":
        # Simple synthetic dataset for smoke tests
        num_classes = int(dataset_cfg.get("num_classes", 10))
        image_size = tuple(dataset_cfg.get("image_size", (3, 32, 32)))
        transform = transforms.ToTensor()
        full_dataset = datasets.FakeData(
            size=int(dataset_cfg.get("size", 2000)),
            image_size=image_size,
            num_classes=num_classes,
            transform=transform,
        )
        val_size = int(0.2 * len(full_dataset))
        test_size = int(0.1 * len(full_dataset))
        train_size = len(full_dataset) - val_size - test_size
        train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])
        input_shape = image_size

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if smoke_test:
        # Clip each split to 256 samples for speed
        train_set = torch.utils.data.Subset(train_set, range(min(256, len(train_set))))
        val_set = torch.utils.data.Subset(val_set, range(min(256, len(val_set))))
        test_set = torch.utils.data.Subset(test_set, range(min(256, len(test_set))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes, input_shape
