"""src/evaluate.py
Aggregates results of all experiment variations, computes comparison
statistics and generates publication-quality figures in .pdf format.
Figures are stored in <results_dir>/figures/ and also listed in the
stdout JSON summary for easy discovery.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

FIG_KWARGS = dict(bbox_inches="tight")


def load_all_results(results_dir: Path) -> List[Dict]:
    results: List[Dict] = []
    for res_file in results_dir.rglob("results.json"):
        with open(res_file) as f:
            results.append(json.load(f))
    if not results:
        raise RuntimeError(f"No results.json files found under {results_dir}")
    return results


def figure_training_loss(all_results: List[Dict], save_dir: Path):
    plt.figure(figsize=(6, 4))
    for res in all_results:
        plt.plot(res["train_loss_history"], label=res["run_id"])
        # Annotate final value
        plt.annotate(f"{res['train_loss_history'][-1]:.3f}",
                     xy=(len(res['train_loss_history']) - 1, res['train_loss_history'][-1]),
                     textcoords="offset points", xytext=(0, 5))
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    fname = save_dir / "training_loss.pdf"
    plt.savefig(fname, **FIG_KWARGS)
    plt.close()
    return str(fname.name)


def figure_accuracy(all_results: List[Dict], save_dir: Path):
    plt.figure(figsize=(6, 4))
    for res in all_results:
        if not res["val_accuracy_history"]:
            continue
        plt.plot(res["val_accuracy_history"], label=res["run_id"])
        plt.annotate(f"{res['val_accuracy_history'][-1]:.3f}",
                     xy=(len(res['val_accuracy_history']) - 1, res['val_accuracy_history'][-1]),
                     textcoords="offset points", xytext=(0, 5))
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Curves")
    plt.legend()
    fname = save_dir / "accuracy.pdf"
    plt.savefig(fname, **FIG_KWARGS)
    plt.close()
    return str(fname.name)


def bar_best_accuracy(all_results: List[Dict], save_dir: Path):
    accs = {r["run_id"]: r["best_val_accuracy"] for r in all_results if r["best_val_accuracy"] is not None}
    if not accs:
        return None
    plt.figure(figsize=(6, 4))
    names = list(accs.keys())
    vals = [accs[n] for n in names]
    sns.barplot(x=names, y=vals)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Comparison of Best Accuracies")
    fname = save_dir / "best_val_accuracy.pdf"
    plt.savefig(fname, **FIG_KWARGS)
    plt.close()
    return str(fname.name)


def evaluate(results_dir: Path):
    print("\n===== Aggregating experiment results =====\n")
    all_results = load_all_results(results_dir)

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)

    figure_files: List[str] = []
    figure_files.append(figure_training_loss(all_results, figures_dir))
    fig_acc = figure_accuracy(all_results, figures_dir)
    if fig_acc:
        figure_files.append(fig_acc)
    fig_bar = bar_best_accuracy(all_results, figures_dir)
    if fig_bar:
        figure_files.append(fig_bar)

    # ------------------------------------------------------------------
    # Consolidated comparison table for stdout -------------------------
    # ------------------------------------------------------------------
    df_rows = []
    for r in all_results:
        df_rows.append({
            "run_id": r["run_id"],
            "best_val_accuracy": r["best_val_accuracy"],
            "training_time_sec": r["training_time_sec"],
            "time_to_target_sec": r["time_to_target_sec"],
        })
    df = pd.DataFrame(df_rows)

    comparison = df.to_dict(orient="records")
    summary = {
        "description": "Comparison of OHGW experiment variations",
        "num_runs": len(all_results),
        "figures": figure_files,
        "table": comparison,
    }
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, required=True)
    args = p.parse_args()

    evaluate(Path(args.results_dir).resolve())
