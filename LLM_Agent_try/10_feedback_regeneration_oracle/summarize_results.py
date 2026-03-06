#!/usr/bin/env python3

import argparse
import os

import yaml

from pipeline_utils import dump_json, load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize feedback regeneration evaluation results.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "feedback_regen_config.yaml"),
        help="Path to config yaml.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results_dir = os.path.join(script_dir, config["paths"]["results_dir"])
    expected_prefixes = [ds["output_prefix"] for ds in config["evaluation"]["datasets"]]
    metrics_files = [os.path.join(results_dir, f"{prefix}_metrics.json") for prefix in expected_prefixes]
    missing = [path for path in metrics_files if not os.path.exists(path)]
    if missing:
        raise RuntimeError(f"Missing metrics files: {missing}")

    summary = {
        "experiment": "10_feedback_regeneration_oracle",
        "results": {},
    }

    for path in sorted(metrics_files):
        data = load_json(path)
        name = os.path.basename(path).replace("_metrics.json", "")
        base = data["baseline_round0_metrics"]
        final = data["final_after_feedback_metrics"]
        summary["results"][name] = {
            "dataset_name": data["dataset_name"],
            "num_samples": data["num_samples"],
            "baseline": {
                "HR@1": base.get("HR@1"),
                "HR@10": base.get("HR@10"),
                "NDCG@10": base.get("NDCG@10"),
            },
            "final": {
                "HR@1": final.get("HR@1"),
                "HR@10": final.get("HR@10"),
                "NDCG@10": final.get("NDCG@10"),
            },
            "delta_final_minus_baseline": {
                "HR@1": data["delta_final_minus_baseline"].get("HR@1"),
                "HR@10": data["delta_final_minus_baseline"].get("HR@10"),
                "NDCG@10": data["delta_final_minus_baseline"].get("NDCG@10"),
            },
            "feedback_trigger_rate": data["feedback_diagnostics"]["feedback_trigger_rate"],
            "avg_generation_calls_per_sample": data["feedback_diagnostics"]["avg_generation_calls_per_sample"],
        }

    out_path = os.path.join(results_dir, "summary.json")
    dump_json(out_path, summary)
    print(f"[summary] saved: {out_path}")


if __name__ == "__main__":
    main()
