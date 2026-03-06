#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from typing import Dict, List

import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(SCRIPT_DIR, "jobs")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
ONEREC_PYTHON = "/home/kfwang/miniconda3/envs/onerec/bin/python"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data) -> None:
    ensure_dir(os.path.dirname(path))
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage review-retrieval enhanced SFT experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(SCRIPT_DIR, "experiment_config.yaml"),
    )
    parser.add_argument("--allowed_gpus", type=str, default="4,5,6,7")
    parser.add_argument("--poll_seconds", type=int, default=45)
    parser.add_argument("--memory_threshold_mb", type=int, default=1000)
    parser.add_argument("--util_threshold", type=int, default=20)
    return parser.parse_args()


def query_allowed_free_gpus(allowed: List[int], memory_threshold_mb: int, util_threshold: int) -> List[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    free = []
    for line in output.strip().splitlines():
        gpu_idx, mem_used, util = [part.strip() for part in line.split(",")]
        gpu_id = int(gpu_idx)
        if gpu_id not in allowed:
            continue
        if int(mem_used) <= memory_threshold_mb and int(util) <= util_threshold:
            free.append(gpu_id)
    return free


def build_jobs(config_path: str, config: dict) -> List[dict]:
    py = ONEREC_PYTHON if os.path.exists(ONEREC_PYTHON) else sys.executable
    paths_cfg = config["paths"]
    return [
        {
            "name": "prepare_dataset",
            "deps": [],
            "ready_path": os.path.join(SCRIPT_DIR, paths_cfg["data_dir"], "dataset_summary.json"),
            "cmd": [py, "prepare_augmented_sft_data.py", "--config", config_path],
            "log_dir": os.path.join(SCRIPT_DIR, paths_cfg["logs_dir"]),
            "use_gpu": False,
        },
        {
            "name": "train_base_init",
            "deps": ["prepare_dataset"],
            "ready_path": os.path.join(SCRIPT_DIR, paths_cfg["output_root"], "base_init", "best_model"),
            "cmd": [py, "train_augmented_sft.py", "--track_name", "base_init", "--config", config_path],
            "log_dir": os.path.join(SCRIPT_DIR, paths_cfg["logs_dir"]),
            "use_gpu": True,
        },
        {
            "name": "train_strong_init",
            "deps": ["prepare_dataset"],
            "ready_path": os.path.join(SCRIPT_DIR, paths_cfg["output_root"], "strong_init", "best_model"),
            "cmd": [py, "train_augmented_sft.py", "--track_name", "strong_init", "--config", config_path],
            "log_dir": os.path.join(SCRIPT_DIR, paths_cfg["logs_dir"]),
            "use_gpu": True,
        },
        {
            "name": "eval_base_init",
            "deps": ["train_base_init"],
            "ready_path": os.path.join(SCRIPT_DIR, paths_cfg["results_dir"], "base_init_evaluation_metrics.json"),
            "cmd": [py, "evaluate_augmented_sft.py", "--track_name", "base_init", "--config", config_path],
            "log_dir": os.path.join(SCRIPT_DIR, paths_cfg["logs_dir"]),
            "use_gpu": True,
        },
        {
            "name": "eval_strong_init",
            "deps": ["train_strong_init"],
            "ready_path": os.path.join(SCRIPT_DIR, paths_cfg["results_dir"], "strong_init_evaluation_metrics.json"),
            "cmd": [py, "evaluate_augmented_sft.py", "--track_name", "strong_init", "--config", config_path],
            "log_dir": os.path.join(SCRIPT_DIR, paths_cfg["logs_dir"]),
            "use_gpu": True,
        },
    ]


def can_start(job: dict, finished_names: set) -> bool:
    return all(dep in finished_names for dep in job["deps"])


def start_job(job: dict, gpu_id: int | None) -> subprocess.Popen:
    log_dir = job.get("log_dir", LOGS_DIR)
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, f"{job['name']}.log")
    log_file = open(log_path, "a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    proc = subprocess.Popen(
        job["cmd"],
        cwd=SCRIPT_DIR,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    job["gpu_id"] = gpu_id
    job["pid"] = proc.pid
    job["log_path"] = log_path
    job["start_time"] = time.time()
    job["status"] = "running"
    job["process"] = proc
    return proc


def dump_status(running: Dict[str, dict], pending: List[dict], finished: List[dict]) -> None:
    save_json(
        os.path.join(JOBS_DIR, "manager_status.json"),
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "running": [
                {
                    "name": job["name"],
                    "pid": job["pid"],
                    "gpu_id": job.get("gpu_id"),
                    "log_path": job["log_path"],
                    "start_time": job["start_time"],
                }
                for job in running.values()
            ],
            "pending": [{"name": job["name"], "deps": job["deps"]} for job in pending],
            "finished": [
                {
                    "name": job["name"],
                    "gpu_id": job.get("gpu_id"),
                    "exit_code": job["exit_code"],
                    "log_path": job.get("log_path"),
                }
                for job in finished
            ],
        },
    )


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    paths_cfg = config["paths"]

    ensure_dir(JOBS_DIR)
    ensure_dir(os.path.join(SCRIPT_DIR, paths_cfg["logs_dir"]))
    ensure_dir(os.path.join(SCRIPT_DIR, paths_cfg["results_dir"]))

    allowed = [int(x) for x in args.allowed_gpus.split(",") if x.strip()]
    pending = deque(build_jobs(args.config, config))
    running: Dict[str, dict] = {}
    finished: List[dict] = []
    finished_names = set()

    print(f"Allowed GPUs: {allowed}")

    while pending or running:
        free_gpus = query_allowed_free_gpus(allowed, args.memory_threshold_mb, args.util_threshold)
        free_slots = list(free_gpus)

        if pending:
            remaining = deque()
            while pending:
                job = pending.popleft()
                if not can_start(job, finished_names):
                    remaining.append(job)
                    continue
                if job["use_gpu"]:
                    if not free_slots:
                        remaining.append(job)
                        continue
                    gpu_id = free_slots.pop(0)
                else:
                    gpu_id = None
                start_job(job, gpu_id)
                running[job["name"]] = job
                print(f"Started {job['name']} on GPU {gpu_id} (pid={job['pid']})")
            pending = remaining
            dump_status(running, list(pending), finished)

        time.sleep(args.poll_seconds)
        completed_names = []
        for name, job in running.items():
            code = job["process"].poll()
            if code is None:
                continue
            job["exit_code"] = code
            finished.append(job)
            finished_names.add(job["name"])
            completed_names.append(name)
            print(f"Finished {job['name']} exit={code}")

        for name in completed_names:
            del running[name]

        dump_status(running, list(pending), finished)

    print("All jobs completed.")


if __name__ == "__main__":
    main()
