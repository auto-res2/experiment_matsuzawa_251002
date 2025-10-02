import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")  # For non-interactive backends
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

FIGURE_PARAMS = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 12,
    "legend.fontsize": 10,
}
plt.rcParams.update(FIGURE_PARAMS)


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    results = []
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        res_file = run_dir / "results.json"
        if res_file.exists():
            with open(res_file) as f:
                results.append(json.load(f))
    return results


def plot_best_accuracy(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(6, 4))
    sns.barplot(x="run_id", y="best_val_accuracy", data=df)
    plt.ylabel("Best Validation Accuracy")
    plt.xlabel("Run ID")
    plt.ylim(0, 1)
    # Annotate bars
    for idx, row in df.iterrows():
        plt.text(idx, row["best_val_accuracy"] + 0.01, f"{row['best_val_accuracy']*100:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(out_path / "accuracy.pdf", bbox_inches="tight")
    plt.close()


def plot_loss_curves(all_results: List[Dict[str, Any]], out_path: Path):
    for res in all_results:
        run_id = res["run_id"]
        epochs = [m["epoch"] for m in res["epoch_metrics"]]
        train_losses = [m["train_loss"] for m in res["epoch_metrics"]]
        val_losses = [m["val_loss"] for m in res["epoch_metrics"]]
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_losses, label="Train loss")
        plt.plot(epochs, val_losses, label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve - {run_id}")
        # Annotate final values
        plt.text(epochs[-1], val_losses[-1], f"{val_losses[-1]:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path / f"training_loss_{run_id}.pdf", bbox_inches="tight")
        plt.close()


def main(results_dir: str):
    results_path = Path(results_dir)
    all_results = load_results(results_path)
    if len(all_results) == 0:
        raise RuntimeError(f"No results.json found in {results_dir}")

    # Convert to pandas DataFrame for easy handling
    df = pd.DataFrame([
        {
            "run_id": r["run_id"],
            "best_val_accuracy": r["best_val_accuracy"],
            "time_to_threshold": r.get("time_to_threshold"),
            "final_test_accuracy": r.get("final_test_accuracy"),
        }
        for r in all_results
    ])

    # Print numerical comparison to stdout
    comparison = df.to_dict(orient="records")
    print(json.dumps({"comparison": comparison}, indent=2))

    # Create figure output directory
    figs_dir = results_path / "figures"
    figs_dir.mkdir(exist_ok=True)

    # Plot best accuracy bar chart
    plot_best_accuracy(df, figs_dir)

    # Plot loss curves per run
    plot_loss_curves(all_results, figs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment variations.")
    parser.add_argument("--results-dir", type=str, required=True, help="Root directory containing all run subdirs.")
    args = parser.parse_args()
    main(args.results_dir)
