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

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    load_dataset = None
    AutoTokenizer = None

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


class WikiTextDataset(Dataset):
    """WikiText dataset for language modeling tasks."""

    def __init__(self, texts, tokenizer, seq_len: int, noise_fraction: float = 0.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.noise_fraction = noise_fraction

        # Tokenize all texts and concatenate
        all_tokens = []
        for text in texts:
            if text.strip():
                tokens = tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(tokens)

        # Split into chunks of seq_len
        self.examples = []
        for i in range(0, len(all_tokens) - seq_len, seq_len):
            chunk = all_tokens[i:i + seq_len]
            if len(chunk) == seq_len:
                self.examples.append(chunk)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        tokens = self.examples[idx].copy()

        # Apply noise if requested
        if self.noise_fraction > 0.0:
            import random
            num_noise = int(len(tokens) * self.noise_fraction)
            vocab_size = self.tokenizer.vocab_size
            for _ in range(num_noise):
                pos = random.randint(0, len(tokens) - 1)
                tokens[pos] = random.randint(0, vocab_size - 1)

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.ones(len(tokens), dtype=torch.long),
        }


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

    if name == "WIKITEXT-103":
        if load_dataset is None or AutoTokenizer is None:
            raise ImportError("datasets and transformers packages required for WikiText-103")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Load dataset
        hf_split = "train" if split == "train" else "validation"
        raw_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=hf_split)

        # Extract text
        texts = raw_dataset["text"]

        # Optionally limit sample size for smoke tests
        sample_size = ds_cfg.get("sample_size", None)
        if sample_size is not None:
            texts = texts[:sample_size]

        seq_len = int(ds_cfg.get("seq_len", 128))
        noise_fraction = float(ds_cfg.get("noise_fraction", 0.0))

        dataset = WikiTextDataset(texts, tokenizer, seq_len, noise_fraction)
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
