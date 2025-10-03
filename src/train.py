"""src/train.py
Core training script for a single experimental run.
Implements end-to-end training, validation, checkpointing, figure generation and
metrics logging.

Usage (called only by main.py):
    python -m src.train \
        --config-file <path/to/config.yaml> \
        --results-dir <path/to/results_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml

from . import preprocess as pp  # type: ignore
from . import model as mdl  # type: ignore

# ----------------------------- Utility functions ----------------------------- #

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


# --------------------------- One-shot warm-starts --------------------------- #

def _perform_warm_start(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    warm_cfg: Dict[str, Any],
    label_smoothing: float,
) -> None:
    """Apply one-shot warm-start as specified in *warm_cfg*.

    Supported variants
    ------------------
    type == "none"   : Do nothing (baseline)
    type == "random" : Add Gaussian noise (σ) to every parameter once.
    type == "hyper"  : Perform *steps* gradient descent steps on the model
                       parameters using a tiny learning-rate *eta_h* on a
                       single mini-batch (approximating hyper-gradient warm-start).
    """
    warm_type = warm_cfg.get("type", "none").lower()
    if warm_type == "none":
        return

    # Fetch exactly one mini-batch (no shuffling issues because iterator is new)
    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    if warm_type == "random":
        sigma = float(warm_cfg.get("sigma", 0.01))
        with torch.no_grad():
            for p in model.parameters():
                p.add_(sigma * torch.randn_like(p))
        return

    if warm_type == "hyper":
        eta_h = float(warm_cfg.get("eta_h", 1e-3))
        steps = int(warm_cfg.get("steps", 1))
        criterion_ws = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion_ws(outputs, targets)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
            with torch.no_grad():
                for p, g in zip(model.parameters(), grads):
                    p.sub_(eta_h * g)  # SGD step on parameters (cheap proxy for hyper-grad)
        return

    raise ValueError(f"Unknown warm-start type '{warm_type}'")


# --------------------------------- Training --------------------------------- #

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_acc += accuracy(outputs.detach(), targets.detach()) * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
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


# --------------------------------- Figures ---------------------------------- #

def save_training_curves(
    metrics: list[dict[str, float]],
    run_id: str,
    results_dir: str,
) -> None:
    images_dir = Path(results_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    epochs = [m["epoch"] for m in metrics]
    train_losses = [m["train_loss"] for m in metrics]
    val_losses = [m["val_loss"] for m in metrics]
    train_accs = [m["train_acc"] for m in metrics]
    val_accs = [m["val_acc"] for m in metrics]

    sns.set_style("whitegrid")

    # Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.scatter(epochs[-1], val_losses[-1], color="red")
    plt.text(
        epochs[-1],
        val_losses[-1],
        f"{val_losses[-1]:.3f}",
        fontsize=8,
        verticalalignment="bottom",
    )
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Training/Validation Loss ({run_id})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(images_dir / f"training_loss_{run_id}.pdf", bbox_inches="tight")
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_accs, label="train_acc")
    plt.plot(epochs, val_accs, label="val_acc")
    plt.scatter(epochs[-1], val_accs[-1], color="red")
    plt.text(
        epochs[-1],
        val_accs[-1],
        f"{val_accs[-1]*100:.2f}%",
        fontsize=8,
        verticalalignment="bottom",
    )
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"Training/Validation Accuracy ({run_id})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(images_dir / f"accuracy_{run_id}.pdf", bbox_inches="tight")
    plt.close()


# --------------------------------- Main ------------------------------------- #

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

    # ---------------------------------------------------------------------
    # Data pipeline
    # ---------------------------------------------------------------------
    train_loader = pp.get_dataloader(cfg, split="train")
    val_loader = pp.get_dataloader(cfg, split="val")

    # ---------------------------------------------------------------------
    # Model & optimisation
    # ---------------------------------------------------------------------
    model = mdl.create_model(cfg).to(device)

    # Criterion with optional label-smoothing
    label_smoothing = float(cfg.get("training", {}).get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optim_cfg = cfg.get("optimizer", {})
    opt_type = optim_cfg.get("type", "SGD").upper()
    lr = float(optim_cfg.get("lr", 0.01))
    momentum = float(optim_cfg.get("momentum", 0.9))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    if opt_type == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # Default to SGD
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    # ---------------------------------------------------------------------
    # One-shot warm-start (if configured)
    # ---------------------------------------------------------------------
    warm_cfg: Dict[str, Any] = cfg.get("warm_start", {"type": "none"})
    _perform_warm_start(model, optimizer, train_loader, device, warm_cfg, label_smoothing)

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    epochs: int = int(cfg.get("training", {}).get("epochs", 10))
    epoch_metrics: list[dict[str, float]] = []
    best_val_acc = -1.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Book-keeping
        metric_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        epoch_metrics.append(metric_dict)

        # Checkpointing
        ckpt_path = checkpoints_dir / f"epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            ckpt_path,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), run_dir / "best_model.pt")

        # Progress feedback (the CI system captures this)
        print(json.dumps({"run_id": run_id, "epoch": epoch, **metric_dict}), flush=True)

    elapsed = time.time() - start_time

    # Final serialization of metrics
    final_metrics = epoch_metrics[-1]
    results_summary: Dict[str, Any] = {
        "run_id": run_id,
        "final_metrics": final_metrics,
        "epoch_metrics": epoch_metrics,
        "training_time_sec": elapsed,
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as f_res:
        json.dump(results_summary, f_res, indent=2)

    # Generate publication-ready figures
    save_training_curves(epoch_metrics, run_id, args.results_dir)

    # Print the summary to stdout last – required by evaluation harness
    print(json.dumps(results_summary), flush=True)


if __name__ == "__main__":
    main()
