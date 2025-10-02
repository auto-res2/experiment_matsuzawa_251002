"""src/train.py
Updated training loop capable of both classification *and* language-model
experiments (GPT-2 on WikiText-103).  The OHGW warm-start logic and the
optimiser now adapt automatically based on `task_type` specified in the
run-config YAML.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

# Local imports
from .preprocess import get_dataloaders  # noqa: E402
from .model import build_model  # noqa: E402

try:
    from transformers import AutoTokenizer  # type: ignore
except ImportError:  # transformers not required for smoke-tests
    AutoTokenizer = None  # type: ignore


# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Run a single OHGW experiment variation")
    p.add_argument("--run-config", type=str, required=True)
    p.add_argument("--results-dir", type=str, required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Hyper-parameter container --------------------------------------------------
# ---------------------------------------------------------------------------

class HyperParameters(nn.Module):
    """Wrap continuous hyper-parameters so that dL/dψ can be obtained by
    autograd.  Each parameter is a scalar torch.Parameter.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        hp_cfg: Dict = cfg.get("hyperparams", {})
        self.log_lr = nn.Parameter(torch.tensor(float(hp_cfg.get("log_lr", -4.0))))
        self.log_wd = nn.Parameter(torch.tensor(float(hp_cfg.get("log_wd", -4.0))))
        # Momentum is irrelevant for AdamW but kept for search-space consistency
        self.raw_momentum = nn.Parameter(torch.tensor(float(hp_cfg.get("momentum", 0.9)).atanh()))

    # Convenience --------------------------------------------------------
    @property
    def lr(self) -> float:
        return float(10 ** self.log_lr.detach())

    @property
    def weight_decay(self) -> float:
        return float(10 ** self.log_wd.detach())

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
# Task-specific helpers ------------------------------------------------------
# ---------------------------------------------------------------------------

def classification_step(model: nn.Module, criterion: nn.Module, xb: torch.Tensor, yb: torch.Tensor,
                        hp: HyperParameters) -> torch.Tensor:
    logits = model(xb)
    loss = criterion(logits, yb)
    wd_coeff = torch.exp(hp.log_wd)
    if wd_coeff > 0:
        wd_term = torch.zeros([], device=logits.device)
        for p in model.parameters():
            wd_term += (p ** 2).sum()
        loss = loss + wd_coeff * wd_term
    return loss


def lm_step(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Forward + loss for causal language modelling using huggingface models."""
    outputs = model(**batch, labels=batch["input_ids"])
    # outputs.loss already incorporates averaging over tokens
    return outputs.loss


# ---------------------------------------------------------------------------
# Evaluation helpers --------------------------------------------------------
# ---------------------------------------------------------------------------

def evaluate_classification(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                            hp: HyperParameters, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = classification_step(model, criterion, xb, yb, hp)
            total_loss += loss.item() * xb.size(0)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            n += xb.size(0)
    return total_loss / max(1, n), correct / max(1, n)


def evaluate_lm(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = lm_step(model, batch)
            losses.append(loss.detach().cpu())
    return float(torch.tensor(losses).mean())  # average loss (NLL)


# ---------------------------------------------------------------------------
# Main training -------------------------------------------------------------
# ---------------------------------------------------------------------------

def train(cfg: Dict, results_dir: Path):
    run_id: str = cfg["run_id"]
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    task_type: str = cfg.get("task_type", "classification").lower()
    is_lm = task_type == "lm"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build data & model --------------------------------------------------
    model = build_model(cfg).to(device)
    train_loader, val_loader = get_dataloaders(cfg)

    # Hyperparameters container -----------------------------------------
    hp = HyperParameters(cfg).to(device)

    criterion = nn.CrossEntropyLoss() if not is_lm else None

    # ---------------- One-Shot Hyper-Gradient Warm-Start -----------------
    warm_batch: Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]] = next(iter(train_loader))

    if is_lm:
        warm_batch = {k: v.to(device) for k, v in warm_batch.items()}
        loss_warm = lm_step(model, warm_batch)
    else:
        xb, yb = warm_batch  # type: ignore
        xb, yb = xb.to(device), yb.to(device)
        loss_warm = classification_step(model, criterion, xb, yb, hp)

    grads = torch.autograd.grad(loss_warm, list(hp.parameters()), create_graph=False)
    eta_h = float(cfg.get("eta_h", 0.0))
    with torch.no_grad():
        for p, g in zip(hp.parameters(), grads):
            p -= eta_h * g  # hyper-parameter update

    # ---------------- Optimiser -----------------------------------------
    opt_name = cfg.get("optimizer", "sgd").lower()
    if opt_name == "adamw":
        optimiser = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    else:
        optimiser = torch.optim.SGD(model.parameters(), lr=hp.lr, momentum=hp.momentum, weight_decay=hp.weight_decay)

    # ---------------- Training loop -------------------------------------
    n_epochs = int(cfg.get("epochs", 3))
    target_metric = cfg.get("target_metric")  # e.g. perplexity <=30

    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []
    val_acc_hist: List[float] = []  # may remain empty for LM

    start_time = time.time()
    time_to_target = None

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            optimiser.zero_grad(set_to_none=True)
            if is_lm:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = lm_step(model, batch)
            else:
                xb, yb = batch  # type: ignore
                xb, yb = xb.to(device), yb.to(device)
                loss = classification_step(model, criterion, xb, yb, hp)
            loss.backward()
            optimiser.step()
            epoch_losses.append(loss.detach().cpu())

        avg_train_loss = float(torch.tensor(epoch_losses).mean())

        # ---------------- Validation ---------------------------------
        if is_lm:
            val_loss = evaluate_lm(model, val_loader, device)
            val_loss_hist.append(val_loss)
            train_loss_hist.append(avg_train_loss)

            # Perplexity target check
            if target_metric is not None and time_to_target is None and val_loss <= target_metric:
                time_to_target = time.time() - start_time

            print(json.dumps({
                "run_id": run_id,
                "event": "epoch_end",
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
            }))
        else:
            val_loss, val_acc = evaluate_classification(model, val_loader, criterion, hp, device)
            train_loss_hist.append(avg_train_loss)
            val_loss_hist.append(val_loss)
            val_acc_hist.append(val_acc)

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

    # ---------------- Persist --------------------------------------------
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "hyperparams": hp.as_dict(),
    }, run_dir / "checkpoint.pt")

    results = {
        "run_id": run_id,
        "task_type": task_type,
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

    # Human summary --------------------------------------------------------
    summary_lines = [
        f"\n=== Summary [{run_id}] ===",
        f"Task           : {task_type}",
        f"Best Val Acc   : {results['best_val_accuracy'] if val_acc_hist else 'N/A'}",
        f"Training time  : {total_time/60:.2f} min",
        f"Time→target    : {time_to_target if time_to_target is not None else 'N/A'} sec",
        "==========================\n",
    ]
    print("\n".join(summary_lines))


# ---------------------------------------------------------------------------
# Entry-point ---------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    import yaml  # local import only when executed

    with open(args.run_config, "r") as fh:
        cfg = yaml.safe_load(fh)

    train(cfg, Path(args.results_dir).resolve())