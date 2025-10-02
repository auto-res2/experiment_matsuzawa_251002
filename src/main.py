"""src/main.py
Central experiment orchestrator. Reads the YAML configuration file that
lists *all* run variations, schedules them on the available GPUs (one GPU
per subprocess), manages logging tee-ing, and finally invokes evaluate.py
once all runs have finished.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

import yaml
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"


# ---------------------------------------------------------------------------
# Helper: live tee of subprocess output to both file and main stdout ------
# ---------------------------------------------------------------------------

def _reader_thread(pipe, tee_file):
    with pipe:
        for line in iter(pipe.readline, b""):
            decoded = line.decode()
            tee_file.write(decoded)
            tee_file.flush()
            sys.stdout.write(decoded)
            sys.stdout.flush()


def launch_subprocess(cmd: List[str], env: Dict[str, str], stdout_path: Path, stderr_path: Path) -> subprocess.Popen:
    stdout_f = open(stdout_path, "w")
    stderr_f = open(stderr_path, "w")

    # Merge stderr into its own pipe for tee-ing
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    threading.Thread(target=_reader_thread, args=(proc.stdout, stdout_f), daemon=True).start()
    threading.Thread(target=_reader_thread, args=(proc.stderr, stderr_f), daemon=True).start()
    return proc


# ---------------------------------------------------------------------------
# Scheduler that ensures ≤ num_gpus concurrent processes ---------------
# ---------------------------------------------------------------------------

def run_all_experiments(config_path: Path, results_dir: Path):
    with open(config_path, "r") as fh:
        cfg_root = yaml.safe_load(fh)

    experiments: List[Dict] = cfg_root["experiments"]
    if not experiments:
        raise ValueError("No experiments found in config file")

    n_available_gpus = torch.cuda.device_count()
    if n_available_gpus == 0:
        print("WARNING: No GPUs detected – running on CPU.")
        n_available_gpus = 1  # schedule serially on CPU

    print(f"Detected {n_available_gpus} GPUs → launching up to {n_available_gpus} concurrent runs.")

    # Queue of pending experiments -------------------------------------
    exp_queue = queue.Queue()
    for exp in experiments:
        exp_queue.put(exp)

    active: Dict[int, subprocess.Popen] = {}
    gpu_ids = list(range(n_available_gpus))

    while not exp_queue.empty() or active:
        # Launch new processes if GPU free
        while gpu_ids and not exp_queue.empty():
            gpu_id = gpu_ids.pop(0)
            exp_cfg = exp_queue.get()
            run_id = exp_cfg["run_id"]

            run_dir = results_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            # Write run-specific YAML config so train.py can read it.
            run_cfg_path = run_dir / "run_config.yaml"
            with open(run_cfg_path, "w") as fh:
                yaml.safe_dump(exp_cfg, fh)

            stdout_path = run_dir / "stdout.log"
            stderr_path = run_dir / "stderr.log"

            cmd = [
                sys.executable,
                "-m",
                "src.train",
                "--run-config",
                str(run_cfg_path),
                "--results-dir",
                str(results_dir),
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            print(f"[MAIN] Launching run_id={run_id} on GPU {gpu_id} …")
            proc = launch_subprocess(cmd, env, stdout_path, stderr_path)
            active[gpu_id] = proc

        # Poll active processes ---------------------------------------
        finished_gpus = []
        for gid, proc in active.items():
            ret = proc.poll()
            if ret is not None:  # finished
                if ret != 0:
                    print(f"[MAIN] WARNING: run on GPU {gid} exited with code {ret}")
                finished_gpus.append(gid)
        for gid in finished_gpus:
            active.pop(gid)
            gpu_ids.append(gid)  # free GPU
        time.sleep(1)

    # All done → evaluation ---------------------------------------------
    print("\nAll runs finished – launching evaluation …\n")
    subprocess.run([
        sys.executable,
        "-m",
        "src.evaluate",
        "--results-dir",
        str(results_dir),
    ], check=True)


# ---------------------------------------------------------------------------
# CLI entry-point -----------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OHGW Experiments Orchestrator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run the smoke-test configuration")
    group.add_argument("--full-experiment", action="store_true", help="Run the full experiment configuration")
    parser.add_argument("--results-dir", type=str, required=True, help="Where to store outputs, logs, figures …")
    args = parser.parse_args()

    if args.smoke_test:
        config_file = ROOT_DIR / "config" / "smoke_test.yaml"
    else:
        config_file = ROOT_DIR / "config" / "full_experiment.yaml"

    run_all_experiments(config_file, Path(args.results_dir).resolve())
