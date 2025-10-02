"""src/train.py
Core training logic for a single experiment variation.
This script is NOT dataset- or model-specific; any such specifics are
supplied via the YAML run-config that main.py passes in.
It implements the One-Shot Hyper-Gradient Warm-Start (OHGW) procedure
once at the beginning of training, then proceeds with standard epochs.
Results, logs and figures for this single run are written into the
<results_dir>/<run_id>/ sub-folder (created if necessary).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

# Local imports – all generic / placeholder-aware
from .preprocess import get_dataloaders  # noqa: E402
from .model import build_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single OHGW experiment variation")
    parser.add_argument("--run-config", type=str, required=True, help="Path to YAML file describing ONE experiment variation.")
    parser.add_argument("--results-dir", type=str, required=True, help="Root directory where all results are stored")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hyper-parameter container
# ---------------------------------------------------------------------------

class HyperParameters(nn.Module):
    """Wraps continuous hyper-parameters as torch.Parameters so that
    autograd can compute dL/dψ. A config field `hyperparams` must be
    provided in the run-config YAML.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        hcfg: Dict = cfg["hyperparams"]
        # Every hyper-param becomes a learnable scalar ‑- additional ones can be
        # plugged in freely by later experiment variants.
        self.log_lr = nn.Parameter(torch.tensor(float(hcfg.get("log_lr", -1.0))))
        self.log_wd = nn.Parameter(torch.tensor(float(hcfg.get("log_wd", -4.0))))
        # momentum can be negative for centred updates → use tanh to keep |m|<1
        self.raw_momentum = nn.Parameter(torch.tensor(float(hcfg.get("momentum", 0.9)).atanh()))

    # Convenient accessors ----------------------------------------------------
    @property
    def lr(self) -> float:
        return float(10.0 ** self.log_lr.detach())

    @property
    def weight_decay(self) -> float:
        return float(10.0 ** self.log_wd.detach())

    @property
    def momentum(self) -> float:
        return float(self.raw_momentum.tanh().detach())

    def as_dict(self):
        return {
            "log_lr": self.log_lr.detach().cpu().item(),
            "log_wd": self.log_wd.detach().cpu().item(),
            "momentum": self.momentum,
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def compute_loss(model: nn.Module, criterion: nn.Module, x: torch.Tensor, y: torch.Tensor, hp: HyperParameters) -> torch.Tensor:
    """Loss = classification_loss + weight-decay that depends on hp.log_wd."""
    logits = model(x)
    loss = criterion(logits, y)
    if hp.log_wd.requires_grad:
        wd_coeff = torch.exp(hp.log_wd)
        wd = torch.zeros([], device=logits.device)
        for p in model.parameters():
            wd = wd + (p ** 2).sum()
        loss = loss + wd_coeff * wd
    return loss


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, hp: HyperParameters, device: torch.device) -> Tuple[float, float]:
    """Returns (average loss, accuracy)"""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = compute_loss(model, criterion, xb, yb, hp)
            total_loss += loss.item() * xb.size(0)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            n += xb.size(0)
    avg_loss = total_loss / max(1, n)
    acc = correct / max(1, n)
    return avg_loss, acc


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(cfg: Dict, results_dir: Path):
    run_id: str = cfg["run_id"]
    torch.manual_seed(int(cfg.get("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------
    # Build model & data loaders (generic – specifics pluggable later)
    # ---------------------------------------------------------------------
    model = build_model(cfg).to(device)
    train_loader, val_loader = get_dataloaders(cfg)

    # Hyper-parameters (as trainable tensors to obtain hyper-gradient)
    hp = HyperParameters(cfg).to(device)

    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------------------
    # One-Shot Hyper-Gradient Warm-Start (OHGW)
    # ---------------------------------------------------------------------
    warm_inputs, warm_targets = next(iter(train_loader))
    warm_inputs, warm_targets = warm_inputs.to(device), warm_targets.to(device)

    loss_warm = compute_loss(model, criterion, warm_inputs, warm_targets, hp)
    grads = torch.autograd.grad(loss_warm, list(hp.parameters()), create_graph=False)

    eta_h: float = float(cfg.get("eta_h", 1e-3))
    with torch.no_grad():
        for p, g in zip(hp.parameters(), grads):
            p -= eta_h * g  # one hyper-step

    # ---------------------------------------------------------------------
    # Optimiser INITIALISED *after* warm-start so that new hp values apply.
    # ---------------------------------------------------------------------
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=hp.lr,
        momentum=hp.momentum,
        weight_decay=hp.weight_decay,
    )

    # ---------------------------------------------------------------------
    # Training loop (standard supervised classification)
    # ---------------------------------------------------------------------
    n_epochs = int(cfg.get("epochs", 10))
    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []
    val_acc_hist: List[float] = []

    start_time = time.time()
    time_to_target: float | None = None

    target_metric = cfg.get("target_metric")  # e.g. 0.93 accuracy

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = compute_loss(model, criterion, xb, yb, hp)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * xb.size(0)
            n_train += xb.size(0)
        avg_train_loss = epoch_loss / max(1, n_train)

        val_loss, val_acc = evaluate(model, val_loader, criterion, hp, device)

        train_loss_hist.append(avg_train_loss)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        # Trigger wall-clock target metric timer
        if target_metric is not None and time_to_target is None and val_acc >= target_metric:
            time_to_target = time.time() - start_time

        print(json.dumps({
            "run_id": run_id,
            "event": "epoch_end",
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }))

    total_time = time.time() - start_time

    # ---------------------------------------------------------------------
    # Persist results ------------------------------------------------------
    # ---------------------------------------------------------------------
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights – may be useful for downstream analyses ----------
    torch.save({
        "model_state_dict": model.state_dict(),
        "hyperparams": hp.as_dict(),
    }, run_dir / "checkpoint.pt")

    # Prepare metrics dict -------------------------------------------------
    results = {
        "run_id": run_id,
        "hyperparameters": hp.as_dict(),
        "train_loss_history": train_loss_hist,
        "val_loss_history": val_loss_hist,
        "val_accuracy_history": val_acc_hist,
        "best_val_accuracy": max(val_acc_hist) if val_acc_hist else None,
        "training_time_sec": total_time,
        "time_to_target_sec": time_to_target,
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Human-readable summary printed to stdout ----------------------------
    summary_str = (
        f"\n=== Experiment Summary [{run_id}] ===\n"
        f"Best Val Acc: {results['best_val_accuracy']:.4f}\n"
        f"Total time     : {total_time/60:.2f} min\n"
        f"Time→target(≥{target_metric}): {time_to_target if time_to_target is not None else 'N/A'} sec\n"
        f"======================================\n"
    )
    print(summary_str)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    import yaml  # local import to avoid dependency during module import

    with open(args.run_config, "r") as fh:
        cfg = yaml.safe_load(fh)

    results_root = Path(args.results_dir).resolve()
    train(cfg, results_root)
