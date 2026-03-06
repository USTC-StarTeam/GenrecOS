#!/usr/bin/env python3

import argparse
import os

import yaml

from pipeline_utils import dump_json, load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize seq2pat-memory SFT experiment results.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sft_pattern_config.yaml"),
        help="Path to config yaml.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    eval_cfg = config["evaluation"]
    data_cfg = config["data"]
    pattern_cfg = config["pattern_memory"]

    data_dir = os.path.join(script_dir, paths["data_dir"])
    results_dir = os.path.join(script_dir, paths["results_dir"])
    output_dir = os.path.join(script_dir, paths["output_dir"])
    summary_path = os.path.join(results_dir, "summary.json")

    dataset_summary = load_json(os.path.join(data_dir, "dataset_summary.json"))

    results = {}
    for ds in eval_cfg["datasets"]:
        prefix = ds["output_prefix"]
        metrics_path = os.path.join(results_dir, f"{prefix}_metrics.json")
        if not os.path.exists(metrics_path):
            raise RuntimeError(f"Missing metrics file: {metrics_path}")
        metrics = load_json(metrics_path)
        results[prefix] = {
            "dataset_name": ds["name"],
            "test_path": metrics.get("test_path", ds["test_path"]),
            "num_test_samples": metrics["num_test_samples"],
            "HR@1": metrics.get("HR@1"),
            "HR@5": metrics.get("HR@5"),
            "HR@10": metrics.get("HR@10"),
            "NDCG@10": metrics.get("NDCG@10"),
            "exact_title_match": metrics.get("exact_title_match"),
        }

    summary = {
        "experiment": "11_seq2pat_memory_prompt",
        "model_checkpoint": os.path.join(output_dir, "best_model"),
        "data_config": data_cfg,
        "pattern_memory_config": pattern_cfg,
        "dataset_summary": dataset_summary,
        "evaluation_results": results,
    }
    dump_json(summary_path, summary)
    print(f"[summary] saved: {summary_path}")


if __name__ == "__main__":
    main()
