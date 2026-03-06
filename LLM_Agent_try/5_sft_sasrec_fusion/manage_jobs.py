#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from typing import Dict, List

from common import JOBS_DIR, LOGS_DIR, RESULTS_DIR, ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch and monitor SFT+SASRec fusion jobs.")
    parser.add_argument("--cache_tag", type=str, default="sft_best_full")
    parser.add_argument("--max_gpus", type=int, default=4)
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--memory_threshold_mb", type=int, default=1000)
    parser.add_argument("--util_threshold", type=int, default=20)
    return parser.parse_args()


def query_free_gpus(memory_threshold_mb: int, util_threshold: int) -> List[int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    free = []
    for line in output.strip().splitlines():
        index_str, mem_str, util_str = [part.strip() for part in line.split(",")]
        if int(mem_str) <= memory_threshold_mb and int(util_str) <= util_threshold:
            free.append(int(index_str))
    return free


def build_jobs(cache_tag: str) -> List[dict]:
    py = sys.executable
    return [
        {
            "name": "precompute",
            "cmd": [py, "precompute_sft_features.py", "--cache_tag", cache_tag, "--device", "cuda:0"],
        },
        {
            "name": "fixed_eval",
            "cmd": [py, "eval_fixed_fusion_sft.py", "--cache_tag", cache_tag, "--device", "cuda:0", "--run_name", "fixed_fusion_sft"],
        },
        {
            "name": "gate_default",
            "cmd": [py, "train_context_gate_sft.py", "--cache_tag", cache_tag, "--device", "cuda:0", "--run_name", "gate_default"],
        },
        {
            "name": "gate_hr1",
            "cmd": [
                py,
                "train_context_gate_sft.py",
                "--cache_tag",
                cache_tag,
                "--device",
                "cuda:0",
                "--run_name",
                "gate_hr1",
                "--dropout",
                "0.05",
                "--residual_scale",
                "0.12",
                "--label_smoothing",
                "0.0",
                "--lr",
                "3e-4",
            ],
        },
        {
            "name": "gate_topk",
            "cmd": [
                py,
                "train_context_gate_sft.py",
                "--cache_tag",
                cache_tag,
                "--device",
                "cuda:0",
                "--run_name",
                "gate_topk",
                "--dropout",
                "0.12",
                "--residual_scale",
                "0.28",
                "--label_smoothing",
                "0.03",
                "--lr",
                "7e-4",
            ],
        },
    ]


def start_job(job: dict, gpu_id: int) -> subprocess.Popen:
    ensure_dir(LOGS_DIR)
    log_path = os.path.join(LOGS_DIR, f"{job['name']}.log")
    log_file = open(log_path, "a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        job["cmd"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
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


def dump_status(running: Dict[int, dict], pending: List[dict], finished: List[dict]) -> None:
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "running": [
            {
                "name": job["name"],
                "pid": job["pid"],
                "gpu_id": job["gpu_id"],
                "log_path": job["log_path"],
                "start_time": job["start_time"],
                "status": job["status"],
            }
            for job in running.values()
        ],
        "pending": [{"name": job["name"]} for job in pending],
        "finished": [
            {
                "name": job["name"],
                "exit_code": job["exit_code"],
                "gpu_id": job.get("gpu_id"),
                "log_path": job.get("log_path"),
            }
            for job in finished
        ],
    }
    save_json(os.path.join(JOBS_DIR, "manager_status.json"), payload)


def main() -> None:
    args = parse_args()
    ensure_dir(JOBS_DIR)
    ensure_dir(RESULTS_DIR)

    free_gpus = query_free_gpus(args.memory_threshold_mb, args.util_threshold)
    if not free_gpus:
        raise RuntimeError("No free GPUs found.")
    free_gpus = free_gpus[: args.max_gpus]
    print(f"Using GPUs: {free_gpus}")

    pending = deque(build_jobs(args.cache_tag))
    running: Dict[int, dict] = {}
    finished: List[dict] = []

    while pending and len(running) < len(free_gpus):
        gpu_id = free_gpus[len(running)]
        job = pending.popleft()
        start_job(job, gpu_id)
        running[gpu_id] = job
        print(f"Started {job['name']} on GPU {gpu_id} (pid={job['pid']})")

    dump_status(running, list(pending), finished)

    while running or pending:
        time.sleep(args.poll_seconds)
        completed_gpu_ids = []
        for gpu_id, job in running.items():
            proc = job["process"]
            code = proc.poll()
            if code is None:
                continue
            job["exit_code"] = code
            job["status"] = "finished"
            finished.append(job)
            completed_gpu_ids.append(gpu_id)
            print(f"Job finished: {job['name']} on GPU {gpu_id} exit={code}")

        for gpu_id in completed_gpu_ids:
            del running[gpu_id]
            if pending:
                job = pending.popleft()
                start_job(job, gpu_id)
                running[gpu_id] = job
                print(f"Started {job['name']} on GPU {gpu_id} (pid={job['pid']})")

        dump_status(running, list(pending), finished)

    print("All jobs completed.")


if __name__ == "__main__":
    main()
