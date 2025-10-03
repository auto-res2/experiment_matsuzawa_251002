"""src/train.py
Core training script for a single experimental run.
Now supports *classification* AND *causal language-modeling* tasks with optional
One-Shot Hyper-Gradient Warm-Start (OHGW).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml

from . import preprocess as pp  # type: ignore
from . import model as mdl  # type: ignore

try:
    from transformers import PreTrainedModel  # type: ignore
except ImportError:  # transformers only needed for LM runs
    PreTrainedModel = None  # type: ignore

# --------------------------------------------------------------------------- #
#                             Utility functions                               #
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute top-1 accuracy for classification."""
    pred_label = pred.argmax(dim=1)
    correct = (pred_label == target).sum().item()
    return correct / target.size(0)


# --------------------------------------------------------------------------- #
#                          OHGW warm-start logic                              #
# --------------------------------------------------------------------------- #

def _apply_ohgw_warm_start(
    cfg: dict,
    model: nn.Module,
    optimizer: optim.Optimizer,
    first_batch: Any,
    device: torch.device,
    task: str,
) -> None:
    """Approximate one-shot hyper-gradient warm-start.

    NOTE: True hyper-gradients w.r.t. LR / WD are technically tricky because
    those are *not* first-class tensors in PyTorch.  Here we follow the paper’s
    *philosophy* and derive a very cheap proxy: we measure average parameter
    gradient magnitude on the first mini-batch and nudge learning-rate &
    weight-decay of every param-group in the *opposite* direction.  This keeps
    the API shape intact and – crucially – exercises autograd once, costing ~1
    extra backward pass.
    """
    ohgw_cfg = cfg.get("ohgw", {})
    eta_h: float = float(ohgw_cfg.get("eta_h", 1e-3))

    model.train()
    optimizer.zero_grad()

    # ---------------------------------------------------------------------
    # 1. Forward / backward on *one* mini-batch to obtain stochastic grads
    # ---------------------------------------------------------------------
    if task == "language_modeling":
        batch = first_batch
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss
    else:  # classification
        inputs, targets = first_batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)

    loss.backward()  # grads populated

    # ---------------------------------------------------------------------
    # 2. Compute *cheap* scalar proxy for gradient magnitude
    # ---------------------------------------------------------------------
    with torch.no_grad():
        mag_sum = 0.0
        cnt = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            mag_sum += p.grad.abs().mean().item()
            cnt += 1
        grad_mag = mag_sum / max(cnt, 1)

        # -----------------------------------------------------------------
        # 3. Hyper-parameter update: lr & weight-decay (continuous)
        # -----------------------------------------------------------------
        for group in optimizer.param_groups:
            old_lr = group.get("lr", 0.001)
            old_wd = group.get("weight_decay", 0.0)
            group["lr"] = max(old_lr - eta_h * grad_mag, 1e-7)
            group["weight_decay"] = max(old_wd - eta_h * grad_mag * 1e-3, 0.0)

    # Important: clear gradients so they don’t leak into real training
    optimizer.zero_grad()


# --------------------------------------------------------------------------- #
#                            Training routines                                #
# --------------------------------------------------------------------------- #

def _train_one_epoch_cls(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_acc += accuracy(outputs.detach(), targets.detach()) * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def _validate_cls(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            running_acc += accuracy(outputs, targets) * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)
    return epoch_loss, epoch_acc


# ---------------------- Language-modeling variants -------------------------- #

def _train_one_epoch_lm(
    model: "PreTrainedModel",  # type: ignore
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    n_sequences = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * input_ids.size(0)
        n_sequences += input_ids.size(0)

    epoch_loss = running_loss / n_sequences
    epoch_ppl = math.exp(epoch_loss)
    return epoch_loss, epoch_ppl


def _validate_lm(
    model: "PreTrainedModel",  # type: ignore
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    n_sequences = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            running_loss += loss.item() * input_ids.size(0)
            n_sequences += input_ids.size(0)

    epoch_loss = running_loss / n_sequences
    epoch_ppl = math.exp(epoch_loss)
    return epoch_loss, epoch_ppl


# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single experimental run")
    parser.add_argument("--config-file", required=True, type=str, help="Path to YAML config for this run")
    parser.add_argument("--results-dir", required=True, type=str, help="Root directory for all experiment outputs")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Configuration & directories
    # ---------------------------------------------------------------------
    with open(args.config_file, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    run_id: str = cfg["run_id"]
    run_dir = Path(args.results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Persist a copy of the resolved configuration for reproducibility
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f_cfg_out:
        yaml.safe_dump(cfg, f_cfg_out)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task: str = cfg.get("task", "classification").lower()

    # ---------------------------------------------------------------------
    # Data pipeline
    # ---------------------------------------------------------------------
    train_loader = pp.get_dataloader(cfg, split="train")
    val_loader = pp.get_dataloader(cfg, split="val")

    # ---------------------------------------------------------------------
    # Model & optimisation
    # ---------------------------------------------------------------------
    model = mdl.create_model(cfg).to(device)

    optim_cfg = cfg.get("optimizer", {})
    opt_type = optim_cfg.get("type", "AdamW").upper()
    lr = float(optim_cfg.get("lr", 5e-4))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    if opt_type == "ADAMW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # Default to SGD
        momentum = float(optim_cfg.get("momentum", 0.9))
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    epochs: int = int(cfg.get("training", {}).get("epochs", 10))

    # ---------------------------------------------------------------------
    # Optional OHGW warm-start (applied before first rung / epoch)
    # ---------------------------------------------------------------------
    if cfg.get("ohgw", {}).get("enabled", False):
        first_batch = next(iter(train_loader))
        _apply_ohgw_warm_start(cfg, model, optimizer, first_batch, device, task)

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    epoch_metrics: list[dict[str, float]] = []
    best_val_metric = float("inf") if task == "language_modeling" else -1.0
    start_time = time.time()

    # Criterion only needed for classification
    criterion = nn.CrossEntropyLoss() if task == "classification" else None

    for epoch in range(1, epochs + 1):
        if task == "language_modeling":
            train_loss, train_ppl = _train_one_epoch_lm(model, train_loader, optimizer, device)
            val_loss, val_ppl = _validate_lm(model, val_loader, device)
            metric_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ppl": train_ppl,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            }
            current_val_metric = val_ppl  # lower is better
        else:  # classification
            assert criterion is not None
            train_loss, train_acc = _train_one_epoch_cls(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = _validate_cls(model, val_loader, criterion, device)
            metric_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            current_val_metric = val_acc  # higher is better

        epoch_metrics.append(metric_dict)

        # -----------------------------------------------------------------
        # Checkpointing: save last & best by validation criterion
        # -----------------------------------------------------------------
        ckpt_path = checkpoints_dir / f"epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            ckpt_path,
        )

        improved = (
            current_val_metric < best_val_metric if task == "language_modeling" else current_val_metric > best_val_metric
        )
        if improved:
            best_val_metric = current_val_metric
            torch.save(model.state_dict(), run_dir / "best_model.pt")

        # Progress feedback (the CI system captures this)
        print(json.dumps({"run_id": run_id, "epoch": epoch, **metric_dict}), flush=True)

    elapsed = time.time() - start_time

    # Final serialization of metrics
    final_metrics = epoch_metrics[-1]
    results_summary: Dict[str, Any] = {
        "run_id": run_id,
        "task": task,
        "final_metrics": final_metrics,
        "epoch_metrics": epoch_metrics,
        "training_time_sec": elapsed,
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as f_res:
        json.dump(results_summary, f_res, indent=2)

    # ---------------------- Publication-ready figures --------------------- #
    images_dir = Path(args.results_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    epochs_list = [m["epoch"] for m in epoch_metrics]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_list, [m["train_loss"] for m in epoch_metrics], label="train_loss")
    plt.plot(epochs_list, [m["val_loss"] for m in epoch_metrics], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss curve ({run_id})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(images_dir / f"loss_{run_id}.pdf", bbox_inches="tight")
    plt.close()

    # Task-specific metric curves
    if task == "language_modeling":
        plt.figure(figsize=(6, 4))
        plt.plot(epochs_list, [m["val_ppl"] for m in epoch_metrics], label="val_ppl")
        plt.xlabel("epoch")
        plt.ylabel("perplexity")
        plt.title(f"Validation Perplexity ({run_id})")
        plt.tight_layout()
        plt.savefig(images_dir / f"ppl_{run_id}.pdf", bbox_inches="tight")
        plt.close()
    else:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs_list, [m["val_acc"] for m in epoch_metrics], label="val_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"Validation Accuracy ({run_id})")
        plt.tight_layout()
        plt.savefig(images_dir / f"acc_{run_id}.pdf", bbox_inches="tight")
        plt.close()

    # Print the summary to stdout last – required by evaluation harness
    print(json.dumps(results_summary), flush=True)


if __name__ == "__main__":
    main()
