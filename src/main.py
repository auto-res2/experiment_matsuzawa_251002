import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import yaml
import json
import datetime


def tee_subprocess(cmd: List[str], stdout_path: Path, stderr_path: Path):
    """Run subprocess while teeing its stdout / stderr to files and console."""
    with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)
        assert process.stdout is not None and process.stderr is not None
        # Stream stdout
        while True:
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()
            if not stdout_line and not stderr_line and process.poll() is not None:
                break
            if stdout_line:
                sys.stdout.buffer.write(stdout_line)
                out_f.write(stdout_line)
                out_f.flush()
            if stderr_line:
                sys.stderr.buffer.write(stderr_line)
                err_f.write(stderr_line)
                err_f.flush()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)


def launch_train(run_cfg: Dict[str, Any], results_root: Path, smoke_test: bool):
    run_id = run_cfg["run_id"]
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run-specific config so that train.py can read it.
    cfg_path = run_dir / "config.yaml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(run_cfg, f)

    # Build subprocess command
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--config",
        str(cfg_path),
        "--results-dir",
        str(run_dir),
    ]
    if smoke_test:
        cmd.append("--smoke-test")

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    tee_subprocess(cmd, stdout_path, stderr_path)


def run_all(cfg_file: str, results_dir: str, smoke_test: bool):
    with open(cfg_file) as f:
        exp_cfg = yaml.safe_load(f)

    if "experiments" not in exp_cfg:
        raise KeyError("Config YAML must contain 'experiments' list.")

    results_root = Path(results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    description = exp_cfg.get("description", "No description provided.")
    print(
        json.dumps(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "experiment_description": description,
            },
            indent=2,
        )
    )

    for run_cfg in exp_cfg["experiments"]:
        print(json.dumps({"status": "starting", "run_id": run_cfg["run_id"]}))
        launch_train(run_cfg, results_root, smoke_test)
        print(json.dumps({"status": "finished", "run_id": run_cfg["run_id"]}))

    # After all runs, launch evaluation
    eval_cmd = [sys.executable, "-m", "src.evaluate", "--results-dir", str(results_root)]
    tee_subprocess(eval_cmd, results_root / "evaluate_stdout.log", results_root / "evaluate_stderr.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main orchestrator for OHGW experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run smoke test config.")
    group.add_argument("--full-experiment", action="store_true", help="Run full experiment config.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory where results will be stored.")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent  # project root
    if args.smoke_test:
        cfg_file = root_dir / "config" / "smoke_test.yaml"
    else:
        cfg_file = root_dir / "config" / "full_experiment.yaml"

    run_all(str(cfg_file), args.results_dir, args.smoke_test)
