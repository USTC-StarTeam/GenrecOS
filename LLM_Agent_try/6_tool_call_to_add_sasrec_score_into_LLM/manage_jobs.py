#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from typing import Dict, List

from common import (
    JOBS_DIR,
    LOGS_DIR,
    ensure_dir,
    evaluation_result_path,
    get_track_best_model_dir,
    teacher_summary_path,
    save_json,
)


ONEREC_PYTHON = "/home/kfwang/miniconda3/envs/onerec/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch tool-call SFT experiments on GPUs 4/5/6/7.")
    parser.add_argument("--allowed_gpus", type=str, default="4,5,6,7")
    parser.add_argument("--poll_seconds", type=int, default=45)
    parser.add_argument("--memory_threshold_mb", type=int, default=1000)
    parser.add_argument("--util_threshold", type=int, default=20)
    return parser.parse_args()


def query_allowed_free_gpus(allowed: List[int], memory_threshold_mb: int, util_threshold: int) -> List[int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    free = []
    for line in output.strip().splitlines():
        gpu_idx, mem_used, util = [part.strip() for part in line.split(",")]
        gpu_id = int(gpu_idx)
        if gpu_id not in allowed:
            continue
        if int(mem_used) <= memory_threshold_mb and int(util) <= util_threshold:
            free.append(gpu_id)
    return free


def build_jobs() -> List[dict]:
    py = ONEREC_PYTHON if os.path.exists(ONEREC_PYTHON) else sys.executable
    return [
        {
            "name": "prepare_pre_sft",
            "deps": [],
            "ready_path": teacher_summary_path("pre_sft"),
            "cmd": [py, "prepare_tool_sft_data.py", "--track_name", "pre_sft", "--device", "cuda:0", "--force_recompute"],
        },
        {
            "name": "prepare_post_sft",
            "deps": [],
            "ready_path": teacher_summary_path("post_sft"),
            "cmd": [py, "prepare_tool_sft_data.py", "--track_name", "post_sft", "--device", "cuda:0", "--force_recompute"],
        },
        {
            "name": "train_pre_sft",
            "deps": ["prepare_pre_sft"],
            "ready_path": get_track_best_model_dir("pre_sft"),
            "cmd": [py, "train_tool_sft.py", "--track_name", "pre_sft"],
        },
        {
            "name": "train_post_sft",
            "deps": ["prepare_post_sft"],
            "ready_path": get_track_best_model_dir("post_sft"),
            "cmd": [py, "train_tool_sft.py", "--track_name", "post_sft"],
        },
        {
            "name": "eval_pre_sft",
            "deps": ["train_pre_sft"],
            "ready_path": evaluation_result_path("pre_sft"),
            "cmd": [py, "evaluate_tool_fusion.py", "--track_name", "pre_sft", "--device", "cuda:0"],
        },
        {
            "name": "eval_post_sft",
            "deps": ["train_post_sft"],
            "ready_path": evaluation_result_path("post_sft"),
            "cmd": [py, "evaluate_tool_fusion.py", "--track_name", "post_sft", "--device", "cuda:0"],
        },
    ]


def can_start(job: dict, finished_names: set) -> bool:
    return all(dep in finished_names for dep in job["deps"])


def start_job(job: dict, gpu_id: int) -> subprocess.Popen:
    ensure_dir(LOGS_DIR)
    log_path = os.path.join(LOGS_DIR, f"{job['name']}.log")
    log_file = open(log_path, "a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["TOKENIZERS_PARALLELISM"] = "false"
    process = subprocess.Popen(
        job["cmd"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    job["gpu_id"] = gpu_id
    job["pid"] = process.pid
    job["process"] = process
    job["status"] = "running"
    job["log_path"] = log_path
    job["start_time"] = time.time()
    return process


def dump_status(running: Dict[int, dict], pending: List[dict], finished: List[dict]) -> None:
    save_json(
        os.path.join(JOBS_DIR, "manager_status.json"),
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "running": [
                {
                    "name": job["name"],
                    "pid": job["pid"],
                    "gpu_id": job["gpu_id"],
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
    ensure_dir(JOBS_DIR)
    ensure_dir(LOGS_DIR)

    allowed = [int(x) for x in args.allowed_gpus.split(",") if x.strip()]
    free_gpus = query_allowed_free_gpus(allowed, args.memory_threshold_mb, args.util_threshold)
    if not free_gpus:
        raise RuntimeError(f"No free GPUs found in allowed set: {allowed}")

    pending = deque(build_jobs())
    running: Dict[int, dict] = {}
    finished: List[dict] = []
    finished_names = set()

    print(f"Allowed GPUs: {allowed}")
    print(f"Currently free GPUs: {free_gpus}")

    while pending or running:
        free_gpus = query_allowed_free_gpus(allowed, args.memory_threshold_mb, args.util_threshold)
        free_slots = [gpu for gpu in free_gpus if gpu not in running]
        if free_slots and pending:
            remaining = deque()
            while pending:
                job = pending.popleft()
                if free_slots and can_start(job, finished_names):
                    gpu_id = free_slots.pop(0)
                    start_job(job, gpu_id)
                    running[gpu_id] = job
                    print(f"Started {job['name']} on GPU {gpu_id} (pid={job['pid']})")
                else:
                    remaining.append(job)
            pending = remaining
            dump_status(running, list(pending), finished)

        time.sleep(args.poll_seconds)
        finished_gpu_ids = []
        for gpu_id, job in running.items():
            exit_code = job["process"].poll()
            if exit_code is None:
                continue
            job["exit_code"] = exit_code
            finished.append(job)
            finished_names.add(job["name"])
            finished_gpu_ids.append(gpu_id)
            print(f"Finished {job['name']} on GPU {gpu_id} exit={exit_code}")

        for gpu_id in finished_gpu_ids:
            del running[gpu_id]

        dump_status(running, list(pending), finished)

    print("All jobs completed.")


if __name__ == "__main__":
    main()
