"""src/evaluate.py
Aggregates results from all experimental runs stored in `results_dir/*/results.json`,
computes summary statistics and generates comparison figures for both
classification and language-modeling runs.
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

    def _score(run_dict: Dict[str, Any]) -> float:
        fm = run_dict["final_metrics"]
        if "val_acc" in fm:
            return fm["val_acc"]  # higher better
        return -fm["val_ppl"]  # lower ppl is better so negate

    best_run = max(results, key=_score)
    summary["best_run_id"] = best_run["run_id"]
    summary["best_metric"] = best_run["final_metrics"].get("val_acc", best_run["final_metrics"].get("val_ppl"))
    return summary


# -------------------------------- Figures ----------------------------------- #

def save_comparison_figures(results: List[Dict[str, Any]], results_dir: str) -> None:
    images_dir = Path(results_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    run_ids = [r["run_id"] for r in results]

    if "val_acc" in results[0]["final_metrics"]:  # classification
        val_metrics = [r["final_metrics"]["val_acc"] for r in results]
        ylabel = "validation accuracy"
    else:
        val_metrics = [r["final_metrics"]["val_ppl"] for r in results]
        ylabel = "validation perplexity"

    plt.figure(figsize=(6, 4))
    sns.barplot(x=run_ids, y=val_metrics, palette="viridis")
    for idx, val in enumerate(val_metrics):
        if ylabel.endswith("accuracy"):
            txt = f"{val*100:.2f}%"
        else:
            txt = f"{val:.2f}"
        plt.text(idx, val, txt, ha="center", va="bottom", fontsize=8)
    plt.ylabel(ylabel)
    plt.title(f"Final {ylabel.capitalize()} Across Runs")
    plt.tight_layout()
    plt.savefig(images_dir / "comparison.pdf", bbox_inches="tight")
    plt.close()


# --------------------------------- Main ------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare experimental runs")
    parser.add_argument("--results-dir", required=True, type=str)
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Load results.json from every run directory
    # ---------------------------------------------------------------------
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
