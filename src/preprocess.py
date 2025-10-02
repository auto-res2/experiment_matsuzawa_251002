"""src/preprocess.py
Dataset loading & preprocessing utilities.  Supports two modes:
1.  `dummy`   – deterministic random tensors (used by smoke-tests)
2.  `wikitext103` – word-level language modelling dataset prepared for
    GPT-2-style causal LM training (sequence length configurable).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Optional heavy deps – imported lazily so smoke-tests remain cheap
try:
    from datasets import load_dataset  # type: ignore
    from transformers import GPT2TokenizerFast  # type: ignore
except ImportError:  # pragma: no cover – not required for dummy tests
    load_dataset = None  # type: ignore
    GPT2TokenizerFast = None  # type: ignore


# ---------------------------------------------------------------------------
# Dummy dataset for smoke-tests ---------------------------------------------
# ---------------------------------------------------------------------------


class RandomTensorDataset(Dataset):
    """Deterministic pseudo-random dataset suitable for pipeline tests."""

    def __init__(self, length: int, input_shape: List[int], num_classes: int):
        super().__init__()
        generator = torch.Generator().manual_seed(42)
        self.data = torch.randn(length, *input_shape, generator=generator)
        self.targets = torch.randint(0, num_classes, (length,), generator=generator)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Helper – build LM TensorDataset -------------------------------------------
# ---------------------------------------------------------------------------

def _build_lm_tensor_dataset(tokenizer, texts: List[str], seq_len: int) -> TensorDataset:
    """Concatenate texts -> chunk into seq_len blocks -> TensorDataset."""
    encodings = tokenizer(texts, return_attention_mask=False, add_special_tokens=False)["input_ids"]
    flat_ids = [token for sub in encodings for token in sub]
    total_len = (len(flat_ids) // seq_len) * seq_len
    if total_len == 0:
        raise ValueError("Not enough tokens to build even a single block")
    flat_ids = flat_ids[:total_len]
    tensor = torch.tensor(flat_ids, dtype=torch.long)
    blocks = tensor.view(-1, seq_len)  # (num_blocks, seq_len)
    return TensorDataset(blocks)


# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def get_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    dataset_name = cfg.get("dataset_name", "dummy").lower()
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 2))

    # ---------------- Dummy --------------------------------------------------
    if dataset_name in {"dummy", "random"}:
        input_shape = cfg.get("input_shape", [1, 28, 28])
        num_classes = int(cfg.get("num_classes", 10))
        length = int(cfg.get("dummy_length", 1024))
        train_ds = RandomTensorDataset(length, input_shape, num_classes)
        val_ds = RandomTensorDataset(length // 4, input_shape, num_classes)
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        )

    # ---------------- WikiText-103 -----------------------------------------
    if dataset_name == "wikitext103":
        if load_dataset is None or GPT2TokenizerFast is None:
            raise RuntimeError("`datasets` and `transformers` packages must be installed for WikiText-103")

        seq_len = int(cfg.get("seq_len", 1024))
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.model_max_length = seq_len
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

        # Load raw dataset ------------------------------------------------
        raw = load_dataset("wikitext", "wikitext-103-raw-v1", trust_remote_code=True)
        train_texts = raw["train"]["text"]
        val_texts = raw["validation"]["text"]

        train_ds = _build_lm_tensor_dataset(tokenizer, train_texts, seq_len)
        val_ds = _build_lm_tensor_dataset(tokenizer, val_texts, seq_len)

        # Collate fn converts tensor batch -> dict required by HF causal LM
        def collate_fn(batch):
            input_ids = torch.stack([item[0] for item in batch])  # (B, seq_len)
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        return train_loader, val_loader

    # ---------------- Unknown dataset --------------------------------------
    raise NotImplementedError(f"Unknown dataset_name='{dataset_name}'. Implement in preprocess.py.")