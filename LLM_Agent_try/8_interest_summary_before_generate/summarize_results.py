#!/usr/bin/env python3

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize common4385 metrics for interest-summary experiment.")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def load_metric(path: str) -> dict:
    return json.load(open(path, "r", encoding="utf-8"))


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir

    base_raw = load_metric(os.path.join(results_dir, "base_raw_common4385_metrics.json"))
    base_interest = load_metric(os.path.join(results_dir, "base_interest_common4385_metrics.json"))
    strong_raw = load_metric(os.path.join(results_dir, "strong_raw_common4385_metrics.json"))
    strong_interest = load_metric(os.path.join(results_dir, "strong_interest_common4385_metrics.json"))

    def delta(a: dict, b: dict):
        return {
            "HR@1": b["HR@1"] - a["HR@1"],
            "HR@10": b["HR@10"] - a["HR@10"],
            "NDCG@10": b["NDCG@10"] - a["NDCG@10"],
        }

    summary = {
        "test_set": "common_4385_aligned",
        "base_raw": {k: base_raw[k] for k in ["HR@1", "HR@10", "NDCG@10", "num_test_samples"]},
        "base_interest": {k: base_interest[k] for k in ["HR@1", "HR@10", "NDCG@10", "num_test_samples"]},
        "strong_raw": {k: strong_raw[k] for k in ["HR@1", "HR@10", "NDCG@10", "num_test_samples"]},
        "strong_interest": {k: strong_interest[k] for k in ["HR@1", "HR@10", "NDCG@10", "num_test_samples"]},
        "delta_base_interest_minus_raw": delta(base_raw, base_interest),
        "delta_strong_interest_minus_raw": delta(strong_raw, strong_interest),
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

