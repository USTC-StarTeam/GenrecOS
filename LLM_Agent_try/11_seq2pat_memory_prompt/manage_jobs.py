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
    parser = argparse.ArgumentParser(description="Manage 11_ seq2pat memory experiment jobs.")
    parser.add_argument("--config", type=str, default=os.path.join(SCRIPT_DIR, "sft_pattern_config.yaml"))
    parser.add_argument("--allowed_gpus", type=str, default="4,5,6,7")
    parser.add_argument("--poll_seconds", type=int, default=20)
    parser.add_argument("--memory_threshold_mb", type=int, default=1200)
    parser.add_argument("--util_threshold", type=int, default=20)
    parser.add_argument("--max_users", type=int, default=0, help="smoke mode only")
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


def build_jobs(config_path: str, cfg: dict, max_users: int) -> List[dict]:
    py = ONEREC_PYTHON if os.path.exists(ONEREC_PYTHON) else sys.executable
    paths = cfg["paths"]
    data_dir = os.path.join(SCRIPT_DIR, paths["data_dir"])
    output_dir = os.path.join(SCRIPT_DIR, paths["output_dir"])
    results_dir = os.path.join(SCRIPT_DIR, paths["results_dir"])
    logs_dir = os.path.join(SCRIPT_DIR, "logs")

    prepare_cmd = [py, "prepare_sft_data_with_pattern_memory.py", "--config", config_path]
    if max_users > 0:
        prepare_cmd.extend(["--max_users", str(max_users)])

    return [
        {
            "name": "prepare_data_with_pattern_memory",
            "deps": [],
            "cmd": prepare_cmd,
            "log_dir": logs_dir,
            "use_gpu": False,
        },
        {
            "name": "build_common4385_test",
            "deps": ["prepare_data_with_pattern_memory"],
            "cmd": [
                py,
                "build_common_4385_eval_set.py",
                "--input_test_path",
                os.path.join(data_dir, "test.jsonl"),
                "--output_test_path",
                os.path.join(data_dir, "test_common_4385.jsonl"),
                "--sasrec_test_path",
                os.path.join(SCRIPT_DIR, cfg["paths"]["sasrec_test_path"]),
                "--item_mapping_path",
                os.path.join(SCRIPT_DIR, cfg["paths"]["item_mapping_path"]),
                "--summary_path",
                os.path.join(data_dir, "test_common_4385.summary.json"),
            ],
            "log_dir": logs_dir,
            "use_gpu": False,
        },
        {
            "name": "train_sft_pattern",
            "deps": ["prepare_data_with_pattern_memory"],
            "cmd": [py, "train_full_sft_pattern.py", "--config", config_path],
            "log_dir": logs_dir,
            "use_gpu": True,
        },
        {
            "name": "eval_pattern_test4548",
            "deps": ["train_sft_pattern"],
            "cmd": [
                py,
                "evaluate_full_sft_pattern.py",
                "--config",
                config_path,
                "--checkpoint_path",
                os.path.join(output_dir, "best_model"),
                "--test_path",
                os.path.join(data_dir, "test.jsonl"),
                "--output_prefix",
                "pattern_test4548",
            ],
            "log_dir": logs_dir,
            "use_gpu": True,
        },
        {
            "name": "eval_pattern_common4385",
            "deps": ["train_sft_pattern", "build_common4385_test"],
            "cmd": [
                py,
                "evaluate_full_sft_pattern.py",
                "--config",
                config_path,
                "--checkpoint_path",
                os.path.join(output_dir, "best_model"),
                "--test_path",
                os.path.join(data_dir, "test_common_4385.jsonl"),
                "--output_prefix",
                "pattern_common4385",
            ],
            "log_dir": logs_dir,
            "use_gpu": True,
        },
        {
            "name": "summarize_results",
            "deps": ["eval_pattern_test4548", "eval_pattern_common4385"],
            "cmd": [py, "summarize_results.py", "--config", config_path],
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
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
    pending = deque(build_jobs(args.config, cfg, args.max_users))
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
                print(
                    f"[start] {job['name']} "
                    f"{'(gpu=' + str(gpu_id) + ')' if gpu_id is not None else '(cpu)'} "
                    f"pid={job['pid']}"
                )
            pending = remaining

        completed = []
        for name, job in list(running.items()):
            ret = job["process"].poll()
            if ret is None:
                continue
            elapsed = time.time() - job["start_time"]
            print(f"[done] {name} exit={ret} elapsed={elapsed/60:.1f}m")
            job["exit_code"] = int(ret)
            finished.append(job)
            completed.append(name)
            if ret == 0:
                finished_names.add(name)
            else:
                dump_status(status_path, running, list(pending), finished)
                raise RuntimeError(f"Job failed: {name}, see log: {job['log_path']}")

        for name in completed:
            running.pop(name, None)

        dump_status(status_path, running, list(pending), finished)
        time.sleep(args.poll_seconds)

    print("All jobs finished successfully.")
    print(f"Status written to {status_path}")


if __name__ == "__main__":
    main()
