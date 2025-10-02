import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")
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

#####################################################################
#                        Helper functions                           #
#####################################################################

def _load_results(results_dir: Path) -> List[Dict[str, Any]]:
    results = []
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        res_file = run_dir / "results.json"
        if res_file.exists():
            with open(res_file) as f:
                results.append(json.load(f))
    return results


def _plot_bar(df: pd.DataFrame, value_col: str, ylabel: str, out_path: Path):
    plt.figure(figsize=(6, 4))
    sns.barplot(x="run_id", y=value_col, data=df)
    plt.ylabel(ylabel)
    plt.xlabel("Run ID")
    # Annotate bars
    for idx, row in df.iterrows():
        plt.text(idx, row[value_col] * 1.01, f"{row[value_col]:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(out_path / f"{value_col}.pdf", bbox_inches="tight")
    plt.close()


def _plot_loss_curves(all_results: List[Dict[str, Any]], out_path: Path):
    for res in all_results:
        run_id = res["run_id"]
        epochs = [m["epoch"] for m in res["epoch_metrics"]]

        if "train_ppl" in res["epoch_metrics"][0]:  # language model
            train_metric = [m["train_ppl"] for m in res["epoch_metrics"]]
            val_metric = [m["val_ppl"] for m in res["epoch_metrics"]]
            ylabel = "Perplexity"
        else:  # classification
            train_metric = [m["train_loss"] for m in res["epoch_metrics"]]
            val_metric = [m["val_loss"] for m in res["epoch_metrics"]]
            ylabel = "Loss"

        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_metric, label=f"Train {ylabel}")
        plt.plot(epochs, val_metric, label=f"Val {ylabel}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} Curve - {run_id}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path / f"{run_id}_{ylabel.lower()}.pdf", bbox_inches="tight")
        plt.close()

#####################################################################
#                              Main                                 #
#####################################################################


def main(results_dir: str):
    results_path = Path(results_dir)
    all_results = _load_results(results_path)
    if len(all_results) == 0:
        raise RuntimeError(f"No results.json found in {results_dir}")

    # Build comparison dataframe supporting both task types
    records = []
    for r in all_results:
        if "best_val_metric" in r:  # new format
            best_val = r["best_val_metric"]
        else:  # fallback
            best_val = r.get("best_val_accuracy", r.get("best_val_ppl", None))

        rec = {
            "run_id": r["run_id"],
            "time_to_threshold": r.get("time_to_threshold"),
        }
        if "final_test_accuracy" in r:
            rec["final_test_accuracy"] = r["final_test_accuracy"]
            rec["best_val_accuracy"] = best_val
        else:
            rec["final_test_ppl"] = r.get("final_test_ppl")
            rec["best_val_ppl"] = best_val
        records.append(rec)

    df = pd.DataFrame(records)
    print(json.dumps({"comparison": records}, indent=2))

    # Figures directory
    figs_dir = results_path / "figures"
    figs_dir.mkdir(exist_ok=True)

    # Plot depending on task type (inspect first row)
    first = records[0]
    if "best_val_accuracy" in first:
        _plot_bar(df, "best_val_accuracy", "Best Val Accuracy", figs_dir)
    else:
        _plot_bar(df, "best_val_ppl", "Best Val Perplexity", figs_dir)

    _plot_loss_curves(all_results, figs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment variations.")
    parser.add_argument("--results-dir", type=str, required=True, help="Root directory containing all run subdirs.")
    args = parser.parse_args()
    main(args.results_dir)
