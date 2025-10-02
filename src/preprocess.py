"""Data loading & preprocessing utilities for both vision and NLP tasks."""

from typing import Tuple, Any, Dict
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, datasets

# Optional heavy imports guarded inside functions to keep import time short

################################################################################
#                              WikiText Dataset                                #
################################################################################

class _WikiTextSequenceDataset(Dataset):
    """Thin wrapper that chunks tokenised text into fixed-length sequences."""

    def __init__(self, texts, tokenizer, seq_len: int = 1024, noise: float = 0.0):
        joined_text = "\n".join(texts)
        tokenised = tokenizer(joined_text, return_tensors="pt").input_ids[0]
        total_len = (tokenised.size(0) // seq_len) * seq_len
        tokenised = tokenised[: total_len]
        self.tokens = tokenised.view(-1, seq_len)  # (num_sequences, seq_len)
        self.noise = noise
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return self.tokens.size(0)

    def __getitem__(self, idx):
        seq = self.tokens[idx].clone()
        if self.noise > 0:
            mask = torch.rand_like(seq.float()) < self.noise
            random_tokens = torch.randint(0, self.vocab_size, size=seq.shape, dtype=torch.long)
            seq[mask] = random_tokens[mask]
        return {"input_ids": seq, "labels": seq.clone()}

################################################################################
#                          Generic get_dataloaders                             #
################################################################################


def get_dataloaders(
    dataset_cfg: Dict[str, Any], *, batch_size: int, num_workers: int, smoke_test: bool
) -> Tuple[DataLoader, DataLoader, DataLoader, int, Any]:
    """Return train/val/test DataLoaders along with `num_classes` and `input_shape`.

    Currently supports:
      • FakeData (vision dummy)
      • Wikitext103 (language modelling)
    """

    dataset_type = dataset_cfg["type"]
    data_root = Path(dataset_cfg.get("root", "./data"))

    ###########################################################################
    #                               FakeData                                 #
    ###########################################################################
    if dataset_type == "FakeData":
        num_classes = dataset_cfg.get("num_classes", 10)
        image_size = dataset_cfg.get("image_size", (3, 32, 32))
        transform = transforms.ToTensor()
        full_dataset = datasets.FakeData(
            size=dataset_cfg.get("size", 2000),
            image_size=image_size,
            num_classes=num_classes,
            transform=transform,
        )
        val_size = int(0.2 * len(full_dataset))
        test_size = int(0.1 * len(full_dataset))
        train_size = len(full_dataset) - val_size - test_size
        train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])
        input_shape = image_size

    ###########################################################################
    #                               WikiText103                               #
    ###########################################################################
    elif dataset_type == "Wikitext103":
        seq_len = dataset_cfg.get("seq_len", 1024)
        noise = float(dataset_cfg.get("noise", 0.0))
        # Lazy import heavy libs here
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        raw_ds = load_dataset("yehzw/wikitext-103", "raw", cache_dir=str(data_root))
        train_texts = raw_ds["train"]["text"]
        val_texts = raw_ds["validation"]["text"]
        test_texts = raw_ds["test"]["text"]

        if smoke_test:
            random.seed(42)
            train_texts = train_texts[:1000]
            val_texts = val_texts[:200]
            test_texts = test_texts[:200]

        train_set = _WikiTextSequenceDataset(train_texts, tokenizer, seq_len, noise=noise)
        val_set = _WikiTextSequenceDataset(val_texts, tokenizer, seq_len, noise=0.0)
        test_set = _WikiTextSequenceDataset(test_texts, tokenizer, seq_len, noise=0.0)

        num_classes = tokenizer.vocab_size  # for compatibility; not used by GPT2
        input_shape = (seq_len,)

    ###########################################################################
    #                            Unknown dataset                              #
    ###########################################################################
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Optional smoke-test subsampling for vision datasets (already done for LM)
    if smoke_test and dataset_type == "FakeData":
        train_set = torch.utils.data.Subset(train_set, range(min(256, len(train_set))))
        val_set = torch.utils.data.Subset(val_set, range(min(256, len(val_set))))
        test_set = torch.utils.data.Subset(test_set, range(min(256, len(test_set))))

    # For language modelling we disable shuffling inside Dataset and instead rely
    # on DataLoader shuffle for each epoch. Pin memory improves GPU input speed.
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_classes, input_shape
