"""src/main.py
Central orchestrator that sequentially executes all run variations defined in
`smoke_test.yaml` or `full_experiment.yaml`, captures their output & error
streams, and finally triggers evaluation.

CLI:
    python -m src.main --smoke-test --results-dir <path>
    python -m src.main --full-experiment --results-dir <path>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, Any

import yaml

# ------------------------- Subprocess streaming utils ----------------------- #

def _stream(pipe, tee_files):
    """Read lines from a pipe and write them to multiple file-like objects."""
    with pipe:
        for line in iter(pipe.readline, b""):
            decoded = line.decode()
            for f in tee_files:
                f.write(decoded)
                f.flush()
            # Always mirror to main process' stdout / stderr
            if tee_files[0].name.endswith("stdout.log"):
                sys.stdout.write(decoded)
                sys.stdout.flush()
            else:
                sys.stderr.write(decoded)
                sys.stderr.flush()


def _run_subprocess(cmd: list[str], stdout_log: Path, stderr_log: Path) -> int:
    """Launch subprocess, tee stdout/err to provided log files *and* console."""
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)

    with open(stdout_log, "w", encoding="utf-8") as fout, open(
        stderr_log, "w", encoding="utf-8") as ferr:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        t_out = threading.Thread(target=_stream, args=(proc.stdout, [fout]))
        t_err = threading.Thread(target=_stream, args=(proc.stderr, [ferr]))
        t_out.start()
        t_err.start()
        t_out.join()
        t_err.join()
        return proc.wait()


# --------------------------------- Main flow -------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment suite")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    grp.add_argument("--full-experiment", action="store_true", help="Run full experiment")
    parser.add_argument("--results-dir", required=True, type=str, help="Output directory")
    args = parser.parse_args()

    root_results_dir = Path(args.results_dir)
    root_results_dir.mkdir(parents=True, exist_ok=True)
    images_dir = root_results_dir / "images"
    images_dir.mkdir(exist_ok=True)

    config_path = (
        Path("config/smoke_test.yaml") if args.smoke_test else Path("config/full_experiment.yaml")
    )

    with open(config_path, "r", encoding="utf-8") as f:
        suite_cfg: Dict[str, Any] = yaml.safe_load(f)

    runs = suite_cfg.get("runs", [])
    if len(runs) == 0:
        raise ValueError("No runs found in configuration file")

    print(f"Running {len(runs)} experiment variations defined in {config_path}\n", flush=True)

    for run_cfg in runs:
        run_id = run_cfg["run_id"]
        print(f"========== Starting run: {run_id} ==========")

        run_dir = root_results_dir / run_id
        run_dir.mkdir(exist_ok=True)

        # Persist individual run config to file so that train.py can read it
        run_cfg_path = run_dir / "config.yaml"
        with open(run_cfg_path, "w", encoding="utf-8") as f_run:
            yaml.safe_dump(run_cfg, f_run)

        cmd = [
            sys.executable,
            "-m",
            "src.train",
            "--config-file",
            str(run_cfg_path),
            "--results-dir",
            str(root_results_dir),
        ]

        stdout_log = run_dir / "stdout.log"
        stderr_log = run_dir / "stderr.log"
        exit_code = _run_subprocess(cmd, stdout_log, stderr_log)
        if exit_code != 0:
            raise RuntimeError(f"Run {run_id} failed with exit code {exit_code}")

    # After all runs, perform evaluation & visualisation
    eval_cmd = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--results-dir",
        str(root_results_dir),
    ]
    exit_code = subprocess.call(eval_cmd)
    if exit_code != 0:
        raise RuntimeError("Evaluation script failed")


if __name__ == "__main__":
    main()
