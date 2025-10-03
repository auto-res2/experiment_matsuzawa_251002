"""src/evaluate.py
Aggregates results from all experimental runs stored in `results_dir/*/results.json`,
computes summary statistics and generates cross-run comparison figures.

Usage:
    python -m src.evaluate --results-dir <path/to/experiments>
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------- Helper utilities ----------------------------- #

def load_run_results(results_path: str) -> Dict[str, Any]:
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------- Metrics ---------------------------------- #

def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "num_runs": len(results),
        "runs": {r["run_id"]: r["final_metrics"] for r in results},
    }
    # Identify best run by validation accuracy
    best_run = max(results, key=lambda r: r["final_metrics"]["val_acc"])
    summary["best_run_id"] = best_run["run_id"]
    summary["best_val_acc"] = best_run["final_metrics"]["val_acc"]
    return summary


# -------------------------------- Figures ----------------------------------- #

def save_comparison_figures(results: List[Dict[str, Any]], results_dir: str) -> None:
    images_dir = Path(results_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart of final validation accuracy
    run_ids = [r["run_id"] for r in results]
    val_accs = [r["final_metrics"]["val_acc"] for r in results]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=run_ids, y=val_accs, palette="viridis")
    for idx, val in enumerate(val_accs):
        plt.text(idx, val + 0.005, f"{val*100:.2f}%", ha="center", va="bottom", fontsize=8)
    plt.ylabel("validation accuracy")
    plt.title("Final Validation Accuracy Across Runs")
    plt.tight_layout()
    plt.savefig(images_dir / "accuracy_comparison.pdf", bbox_inches="tight")
    plt.close()

    # Line graph of validation accuracy over epochs
    plt.figure(figsize=(6, 4))
    for r in results:
        epochs = [m["epoch"] for m in r["epoch_metrics"]]
        val_accs_epoch = [m["val_acc"] for m in r["epoch_metrics"]]
        plt.plot(epochs, val_accs_epoch, label=r["run_id"])
        plt.scatter(epochs[-1], val_accs_epoch[-1])
    plt.xlabel("epoch")
    plt.ylabel("validation accuracy")
    plt.title("Validation Accuracy Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(images_dir / "accuracy_trajectories.pdf", bbox_inches="tight")
    plt.close()


# --------------------------------- Main ------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare experimental runs")
    parser.add_argument("--results-dir", required=True, type=str)
    args = parser.parse_args()

    # --------------------------------------------------------------------------------
    # Load results.json from every run directory
    # --------------------------------------------------------------------------------
    result_files = glob(os.path.join(args.results_dir, "*", "results.json"))
    if len(result_files) == 0:
        raise FileNotFoundError(f"No results.json files found under {args.results_dir}")

    results = [load_run_results(p) for p in sorted(result_files)]

    # Aggregate and summarise
    summary = aggregate_metrics(results)

    # Generate figures
    save_comparison_figures(results, args.results_dir)

    # Final JSON summary to stdout
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
