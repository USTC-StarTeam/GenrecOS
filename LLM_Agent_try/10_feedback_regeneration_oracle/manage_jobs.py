#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from pipeline_utils import dump_json, ensure_dir


@dataclass
class Job:
    name: str
    cmd: List[str]
    deps: List[str] = field(default_factory=list)
    require_gpu: bool = False


@dataclass
class RunningJob:
    job: Job
    process: subprocess.Popen
    log_path: str
    gpu_id: Optional[int]
    start_time: float


def parse_args():
    parser = argparse.ArgumentParser(description="Run and monitor feedback-regeneration experiment jobs.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "feedback_regen_config.yaml"),
        help="Path to config yaml.",
    )
    parser.add_argument("--allowed_gpus", type=str, default=None, help="Comma-separated GPU ids (e.g. 0,1).")
    parser.add_argument("--poll_seconds", type=int, default=None, help="Polling interval.")
    return parser.parse_args()


def build_jobs(script_dir: str, config_path: str, config: dict) -> List[Job]:
    py_exec = sys.executable
    eval_datasets = config["evaluation"]["datasets"]

    jobs: List[Job] = [
        Job(
            name="prepare_data",
            cmd=[py_exec, os.path.join(script_dir, "prepare_feedback_eval_data.py"), "--config", config_path],
            deps=[],
            require_gpu=False,
        )
    ]

    eval_job_names = []
    for ds in eval_datasets:
        job_name = f"eval_{ds['name']}"
        eval_job_names.append(job_name)
        jobs.append(
            Job(
                name=job_name,
                cmd=[
                    py_exec,
                    os.path.join(script_dir, "evaluate_feedback_regeneration.py"),
                    "--config",
                    config_path,
                    "--dataset_name",
                    ds["name"],
                    "--output_prefix",
                    ds["output_prefix"],
                ],
                deps=["prepare_data"],
                require_gpu=True,
            )
        )

    jobs.append(
        Job(
            name="summarize_results",
            cmd=[py_exec, os.path.join(script_dir, "summarize_results.py"), "--config", config_path],
            deps=eval_job_names,
            require_gpu=False,
        )
    )
    return jobs


def launch_job(job: Job, logs_dir: str, gpu_id: Optional[int]) -> RunningJob:
    log_path = os.path.join(logs_dir, f"{job.name}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    process = subprocess.Popen(
        job.cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return RunningJob(
        job=job,
        process=process,
        log_path=log_path,
        gpu_id=gpu_id,
        start_time=time.time(),
    )


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths_cfg = config["paths"]
    runtime_cfg = config.get("runtime", {})
    logs_dir = os.path.join(script_dir, paths_cfg["logs_dir"])
    jobs_dir = os.path.join(script_dir, paths_cfg["jobs_dir"])
    ensure_dir(logs_dir)
    ensure_dir(jobs_dir)

    allowed_gpus_str = args.allowed_gpus or runtime_cfg.get("allowed_gpus", "0")
    allowed_gpus = [int(x.strip()) for x in allowed_gpus_str.split(",") if x.strip()]
    poll_seconds = int(args.poll_seconds or runtime_cfg.get("poll_seconds", 20))

    jobs = build_jobs(script_dir=script_dir, config_path=args.config, config=config)
    pending = {job.name: job for job in jobs}
    running: Dict[str, RunningJob] = {}
    finished = {}
    status_path = os.path.join(jobs_dir, "manager_status.json")

    while pending or running:
        # Check running jobs.
        done_names = []
        for name, rj in running.items():
            ret = rj.process.poll()
            if ret is not None:
                done_names.append(name)
                finished[name] = {
                    "name": name,
                    "gpu_id": rj.gpu_id,
                    "exit_code": ret,
                    "log_path": rj.log_path,
                }
        for name in done_names:
            running.pop(name, None)

        # Try to launch pending jobs.
        for name in list(pending.keys()):
            job = pending[name]
            if any(dep not in finished or finished[dep]["exit_code"] != 0 for dep in job.deps):
                continue

            gpu_id = None
            if job.require_gpu:
                used = {rj.gpu_id for rj in running.values() if rj.gpu_id is not None}
                free = [gid for gid in allowed_gpus if gid not in used]
                if not free:
                    continue
                gpu_id = free[0]

            running[name] = launch_job(job=job, logs_dir=logs_dir, gpu_id=gpu_id)
            pending.pop(name)

        status = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "running": [
                {
                    "name": rj.job.name,
                    "pid": rj.process.pid,
                    "gpu_id": rj.gpu_id,
                    "log_path": rj.log_path,
                    "start_time": rj.start_time,
                }
                for rj in running.values()
            ],
            "pending": [
                {"name": job.name, "deps": job.deps}
                for job in pending.values()
            ],
            "finished": [finished[name] for name in sorted(finished.keys())],
        }
        dump_json(status_path, status)
        time.sleep(poll_seconds)

    final_status = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "running": [],
        "pending": [],
        "finished": [finished[name] for name in sorted(finished.keys())],
    }
    dump_json(status_path, final_status)
    print(f"[manager] done. status -> {status_path}")


if __name__ == "__main__":
    main()
