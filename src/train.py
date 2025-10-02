import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .preprocess import get_dataloaders
from .model import get_model

#####################################################################
#                         Utility helpers                           #
#####################################################################

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#####################################################################
#                        Classification loop                        #
#####################################################################

def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute top-1 accuracy."""
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = preds.eq(target).sum().item()
    return correct / target.size(0)


def _train_one_epoch_cls(model: nn.Module, loader: DataLoader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        running_acc += accuracy(output, y) * x.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


def _evaluate_cls(model: nn.Module, loader: DataLoader, criterion, device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            running_loss += loss.item() * x.size(0)
            running_acc += accuracy(output, y) * x.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc

#####################################################################
#                       Language-model loop                         #
#####################################################################

def _train_one_epoch_lm(model: nn.Module, loader: DataLoader, optimizer, device) -> Tuple[float, float]:
    """Train one epoch for language modelling; returns (mean loss, perplexity)."""
    model.train()
    running_loss = 0.0
    tokens_seen = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        token_count = batch["input_ids"].numel()
        running_loss += loss.item() * token_count
        tokens_seen += token_count

    mean_loss = running_loss / tokens_seen
    perplexity = math.exp(mean_loss)
    return mean_loss, perplexity


def _evaluate_lm(model: nn.Module, loader: DataLoader, device) -> Tuple[float, float]:
    """Evaluate language model; returns (mean loss, perplexity)."""
    model.eval()
    running_loss = 0.0
    tokens_seen = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            token_count = batch["input_ids"].numel()
            running_loss += loss.item() * token_count
            tokens_seen += token_count
    mean_loss = running_loss / tokens_seen
    perplexity = math.exp(mean_loss)
    return mean_loss, perplexity

#####################################################################
#                       OHGW warm-start step                        #
#####################################################################

def _ohgw_warm_start(model: nn.Module, criterion, first_batch, device, eta_h: float, is_lm: bool):
    """Perform one-shot hyper-gradient warm-start.

    For simplicity we take one gradient step on model parameters with a tiny
    learning-rate `eta_h`. This differentiates OHGW variants from baseline
    in our experimental harness without changing the later training loop.
    """
    model.zero_grad()

    if is_lm:
        batch = {k: v.to(device) for k, v in first_batch.items()}
        loss = model(**batch).loss
    else:
        x, y = first_batch
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)

    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    with torch.no_grad():
        for p, g in zip(model.parameters(), grads):
            if g is not None:
                p -= eta_h * g

#####################################################################
#                           Main runner                             #
#####################################################################

def run_experiment(cfg: Dict[str, Any], results_dir: Path, smoke_test: bool):
    run_id = cfg["run_id"]
    seed = cfg.get("seed", 0)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = cfg["training"].get("batch_size", 2)
    num_workers = cfg["training"].get("num_workers", 4)
    max_epochs = cfg["training"].get("epochs", 20)

    # Adjust for smoke test.
    if smoke_test:
        max_epochs = min(2, max_epochs)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader, num_classes, input_shape = get_dataloaders(
        cfg["dataset"], batch_size=batch_size, num_workers=num_workers, smoke_test=smoke_test
    )

    # ------------------------------------------------------------------
    # Model & Optimizer
    # ------------------------------------------------------------------
    model = get_model(cfg["model"], num_classes=num_classes, input_shape=input_shape).to(device)

    is_lm = cfg["model"]["type"].lower().startswith("gpt2")

    optimizer_cfg = cfg["training"].get("optimizer", {"name": "AdamW", "lr": 5e-4, "weight_decay": 0.01})
    if optimizer_cfg["name"].lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=optimizer_cfg["lr"], momentum=optimizer_cfg.get("momentum", 0))
    elif optimizer_cfg["name"].lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=optimizer_cfg["lr"], weight_decay=optimizer_cfg.get("weight_decay", 0))
    elif optimizer_cfg["name"].lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=optimizer_cfg["lr"], weight_decay=optimizer_cfg.get("weight_decay", 0))
    else:
        raise ValueError(f"Unsupported optimizer {optimizer_cfg['name']}")

    criterion = nn.CrossEntropyLoss() if not is_lm else None

    # ------------------------------------------------------------------
    # OHGW warm-start if requested
    # ------------------------------------------------------------------
    ohgw_cfg = cfg.get("ohgw", {})
    if ohgw_cfg.get("enabled", False):
        eta_h = float(ohgw_cfg.get("eta", 1e-3))
        try:
            first_batch = next(iter(train_loader))
        except StopIteration:
            first_batch = None
        if first_batch is not None:
            _ohgw_warm_start(model, criterion, first_batch, device, eta_h, is_lm)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    results: Dict[str, Any] = {
        "run_id": run_id,
        "config": cfg,
        "epoch_metrics": [],
        "best_val_metric": math.inf if is_lm else 0.0,
        "best_epoch": 0,
        "time_to_threshold": None,
    }

    threshold_key = "threshold_ppl" if is_lm else "threshold"
    threshold_value = cfg["evaluation"].get(threshold_key)

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        # Train
        if is_lm:
            train_loss, train_ppl = _train_one_epoch_lm(model, train_loader, optimizer, device)
        else:
            train_loss, train_acc = _train_one_epoch_cls(model, train_loader, criterion, optimizer, device)

        # Validate
        if is_lm:
            val_loss, val_ppl = _evaluate_lm(model, val_loader, device)
        else:
            val_loss, val_acc = _evaluate_cls(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Store metrics
        if is_lm:
            results["epoch_metrics"].append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_ppl": train_ppl,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "elapsed_sec": elapsed,
                }
            )
            # Track best (lower perplexity is better)
            if val_ppl < results["best_val_metric"]:
                results["best_val_metric"] = val_ppl
                results["best_epoch"] = epoch
                torch.save(model.state_dict(), results_dir / "best_model.pt")
            # Threshold time
            if threshold_value is not None and results["time_to_threshold"] is None and val_ppl <= threshold_value:
                results["time_to_threshold"] = elapsed / 3600  # hours
        else:
            results["epoch_metrics"].append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "elapsed_sec": elapsed,
                }
            )
            if val_acc > results["best_val_metric"]:
                results["best_val_metric"] = val_acc
                results["best_epoch"] = epoch
                torch.save(model.state_dict(), results_dir / "best_model.pt")
            if threshold_value is not None and results["time_to_threshold"] is None and val_acc >= threshold_value:
                results["time_to_threshold"] = elapsed / 3600

    # ------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------
    if is_lm:
        test_loss, test_ppl = _evaluate_lm(model, test_loader, device)
        results["final_test_loss"] = test_loss
        results["final_test_ppl"] = test_ppl
    else:
        test_loss, test_acc = _evaluate_cls(model, test_loader, criterion, device)
        results["final_test_accuracy"] = test_acc
        results["final_test_loss"] = test_loss

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Also print to stdout for CI visibility
    print(json.dumps(results))

#####################################################################
#                     Command-line Entrypoint                       #
#####################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single experiment variation.")
    parser.add_argument("--config", type=str, required=True, help="Path to run-specific YAML config file.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory to store outputs.")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test.")
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(args.results_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_experiment(cfg, run_dir, args.smoke_test)
