#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from typing import Dict, List

import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONEREC_PYTHON = "/home/kfwang/miniconda3/envs/onerec/bin/python"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage 9_ SFT-with-interest experiment jobs.")
    parser.add_argument("--config", type=str, default=os.path.join(SCRIPT_DIR, "sft_interest_config.yaml"))
    parser.add_argument("--allowed_gpus", type=str, default="0,1,2,3")
    parser.add_argument("--poll_seconds", type=int, default=20)
    parser.add_argument("--memory_threshold_mb", type=int, default=1200)
    parser.add_argument("--util_threshold", type=int, default=20)
    return parser.parse_args()


def query_free_gpus(allowed: List[int], memory_threshold_mb: int, util_threshold: int) -> List[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    free = []
    for line in output.strip().splitlines():
        idx, mem, util = [x.strip() for x in line.split(",")]
        gpu = int(idx)
        if gpu not in allowed:
            continue
        if int(mem) <= memory_threshold_mb and int(util) <= util_threshold:
            free.append(gpu)
    return free


def build_jobs(config_path: str, cfg: dict) -> List[dict]:
    py = ONEREC_PYTHON if os.path.exists(ONEREC_PYTHON) else sys.executable
    paths = cfg["paths"]
    data_dir = os.path.join(SCRIPT_DIR, paths["data_dir"])
    output_dir = os.path.join(SCRIPT_DIR, paths["output_dir"])
    results_dir = os.path.join(SCRIPT_DIR, paths["results_dir"])
    logs_dir = os.path.join(SCRIPT_DIR, "logs")

    return [
        {
            "name": "prepare_data_with_interest",
            "deps": [],
            "cmd": [py, "prepare_sft_data_with_interest.py", "--config", config_path],
            "log_dir": logs_dir,
            "use_gpu": False,
        },
        {
            "name": "train_sft_interest",
            "deps": ["prepare_data_with_interest"],
            "cmd": [py, "train_full_sft_interest.py", "--config", config_path],
            "log_dir": logs_dir,
            "use_gpu": True,
        },
        {
            "name": "eval_interest_test4548",
            "deps": ["train_sft_interest"],
            "cmd": [
                py,
                "evaluate_full_sft_interest.py",
                "--config",
                config_path,
                "--checkpoint_path",
                os.path.join(output_dir, "best_model"),
                "--test_path",
                os.path.join(data_dir, "test.jsonl"),
                "--output_prefix",
                "interest_test4548",
            ],
            "log_dir": logs_dir,
            "use_gpu": True,
        },
        {
            "name": "eval_raw_test4548",
            "deps": ["train_sft_interest"],
            "cmd": [
                py,
                "evaluate_full_sft_interest.py",
                "--config",
                config_path,
                "--checkpoint_path",
                os.path.join(output_dir, "best_model"),
                "--test_path",
                os.path.join(data_dir, "test_raw.jsonl"),
                "--output_prefix",
                "raw_test4548",
            ],
            "log_dir": logs_dir,
            "use_gpu": True,
        },
        {
            "name": "eval_interest_common4385",
            "deps": ["train_sft_interest"],
            "cmd": [
                py,
                "evaluate_full_sft_interest.py",
                "--config",
                config_path,
                "--checkpoint_path",
                os.path.join(output_dir, "best_model"),
                "--test_path",
                os.path.join(data_dir, "test_common_4385.jsonl"),
                "--output_prefix",
                "interest_common4385",
            ],
            "log_dir": logs_dir,
            "use_gpu": True,
        },
        {
            "name": "eval_raw_common4385",
            "deps": ["train_sft_interest"],
            "cmd": [
                py,
                "evaluate_full_sft_interest.py",
                "--config",
                config_path,
                "--checkpoint_path",
                os.path.join(output_dir, "best_model"),
                "--test_path",
                os.path.join(data_dir, "test_raw_common_4385.jsonl"),
                "--output_prefix",
                "raw_common4385",
            ],
            "log_dir": logs_dir,
            "use_gpu": True,
        },
        {
            "name": "summarize_results",
            "deps": [
                "eval_interest_test4548",
                "eval_raw_test4548",
                "eval_interest_common4385",
                "eval_raw_common4385",
            ],
            "cmd": [
                py,
                "summarize_results.py",
                "--results_dir",
                results_dir,
                "--output_path",
                os.path.join(results_dir, "summary.json"),
            ],
            "log_dir": logs_dir,
            "use_gpu": False,
        },
    ]


def can_start(job: dict, finished_names: set) -> bool:
    return all(dep in finished_names for dep in job["deps"])


def start_job(job: dict, gpu_id: int | None) -> None:
    ensure_dir(job["log_dir"])
    log_path = os.path.join(job["log_dir"], f"{job['name']}.log")
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
    job["process"] = proc
    job["pid"] = proc.pid
    job["gpu_id"] = gpu_id
    job["log_path"] = log_path
    job["start_time"] = time.time()
    job["status"] = "running"


def dump_status(status_path: str, running: Dict[str, dict], pending: List[dict], finished: List[dict]) -> None:
    save_json(
        status_path,
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
            "pending": [{"name": j["name"], "deps": j["deps"]} for j in pending],
            "finished": [
                {
                    "name": j["name"],
                    "gpu_id": j.get("gpu_id"),
                    "exit_code": j["exit_code"],
                    "log_path": j.get("log_path"),
                }
                for j in finished
            ],
        },
    )


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    paths = cfg["paths"]

    jobs_dir = os.path.join(SCRIPT_DIR, paths["jobs_dir"])
    ensure_dir(jobs_dir)
    status_path = os.path.join(jobs_dir, "manager_status.json")

    allowed = [int(x) for x in args.allowed_gpus.split(",") if x.strip()]
    pending = deque(build_jobs(args.config, cfg))
    running: Dict[str, dict] = {}
    finished: List[dict] = []
    finished_names = set()

    print(f"Allowed GPUs: {allowed}")

    while pending or running:
        free = query_free_gpus(allowed, args.memory_threshold_mb, args.util_threshold)
        free_slots = list(free)

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
                print(f"Started {job['name']} on GPU {gpu_id}, pid={job['pid']}")
            pending = remaining
            dump_status(status_path, running, list(pending), finished)

        time.sleep(args.poll_seconds)
        completed = []
        for name, job in running.items():
            code = job["process"].poll()
            if code is None:
                continue
            job["exit_code"] = code
            finished.append(job)
            finished_names.add(name)
            completed.append(name)
            print(f"Finished {name}, exit={code}")

        for name in completed:
            del running[name]
        dump_status(status_path, running, list(pending), finished)

    print("All jobs completed.")


if __name__ == "__main__":
    main()
