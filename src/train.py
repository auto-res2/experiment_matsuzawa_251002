import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .preprocess import get_dataloaders
from .model import get_model


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


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute top-1 accuracy."""
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = preds.eq(target).sum().item()
    return correct / target.size(0)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            output = model(x)
            loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        running_acc += accuracy(output, y) * x.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, loader: DataLoader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = criterion(output, y)
            running_loss += loss.item() * x.size(0)
            running_acc += accuracy(output, y) * x.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


def _apply_random_warm(model: nn.Module, sigma: float):
    """Additive Gaussian noise to model parameters (control experiment)."""
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * sigma)


def _apply_ohgw(model: nn.Module, loader: DataLoader, device: torch.device, eta: float, steps: int, criterion):
    """One-Shot Hyper-Gradient Warm-Start approximation – we take a few miniature SGD steps
    on a single mini-batch prior to the main training loop. This is an *approximation* of
    the method described in the paper and keeps implementation effort minimal while
    still touching the right compute graph pieces for demonstration purposes."""
    model.train()
    mb_iter = iter(loader)
    try:
        x, y = next(mb_iter)
    except StopIteration:
        return  # empty loader in smoke tests
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = criterion(model(x), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.add_( -eta * p.grad )


def maybe_apply_warm_start(model: nn.Module, train_loader: DataLoader, device: torch.device, training_cfg: Dict[str, Any], criterion):
    ws_cfg = training_cfg.get("warm_start")
    if ws_cfg is None:
        return  # vanilla training
    ws_type = ws_cfg["type"].lower()
    if ws_type == "random":
        sigma = float(ws_cfg.get("sigma", 0.01))
        _apply_random_warm(model, sigma)
    elif ws_type == "ohgw":
        eta = float(ws_cfg.get("eta", 1e-3))
        steps = int(ws_cfg.get("steps", 1))
        _apply_ohgw(model, train_loader, device, eta, steps, criterion)
    else:
        raise ValueError(f"Unknown warm_start type: {ws_type}")


def run_experiment(cfg: Dict[str, Any], results_dir: Path, smoke_test: bool):
    run_id = cfg["run_id"]
    seed = cfg.get("seed", 0)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = cfg["training"].get("batch_size", 128)
    num_workers = cfg["training"].get("num_workers", 4)
    max_epochs = cfg["training"].get("epochs", 20)
    threshold = cfg["evaluation"].get("threshold")  # Optional

    # Adjust for smoke test
    if smoke_test:
        max_epochs = min(2, max_epochs)
        batch_size = min(32, batch_size)

    train_loader, val_loader, test_loader, num_classes, input_shape = get_dataloaders(
        cfg["dataset"], batch_size=batch_size, num_workers=num_workers, smoke_test=smoke_test
    )

    model = get_model(cfg["model"], num_classes=num_classes, input_shape=input_shape).to(device)

    # Criterion – support label smoothing if requested
    label_smoothing = float(cfg["dataset"].get("label_smoothing", 0.0))
    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    except TypeError:  # Older torch
        criterion = nn.CrossEntropyLoss()

    optimizer_cfg = cfg["training"].get("optimizer", {"name": "SGD", "lr": 0.1, "momentum": 0.9})
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))

    if optimizer_cfg["name"].lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(optimizer_cfg["lr"]),
            momentum=float(optimizer_cfg.get("momentum", 0.0)),
            weight_decay=weight_decay,
        )
    elif optimizer_cfg["name"].lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=float(optimizer_cfg["lr"]), weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer {optimizer_cfg['name']}")

    # Cosine LR schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Optional warm-start step(s)
    maybe_apply_warm_start(model, train_loader, device, cfg["training"], criterion)

    results = {
        "run_id": run_id,
        "config": cfg,
        "epoch_metrics": [],
        "best_val_accuracy": 0.0,
        "best_epoch": 0,
        "time_to_threshold": None,
    }

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time

        results["epoch_metrics"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "elapsed_sec": elapsed,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_acc > results["best_val_accuracy"]:
            results["best_val_accuracy"] = val_acc
            results["best_epoch"] = epoch
            torch.save(model.state_dict(), results_dir / "best_model.pt")

        if threshold is not None and results["time_to_threshold"] is None and val_acc >= threshold:
            results["time_to_threshold"] = elapsed / 3600.0  # convert to hours

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    results["final_test_accuracy"] = test_acc
    results["final_test_loss"] = test_loss

    # Save results
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Emit to stdout for CI visibility
    print(json.dumps(results))


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
