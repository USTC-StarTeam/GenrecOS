#!/usr/bin/env python3

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize 9_ experiment metrics.")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick(metrics: dict):
    return {
        "HR@1": metrics["HR@1"],
        "HR@10": metrics["HR@10"],
        "NDCG@10": metrics["NDCG@10"],
        "num_test_samples": metrics["num_test_samples"],
    }


def main():
    args = parse_args()
    rd = args.results_dir
    interest_4548 = load_json(os.path.join(rd, "interest_test4548_metrics.json"))
    raw_4548 = load_json(os.path.join(rd, "raw_test4548_metrics.json"))
    interest_common = load_json(os.path.join(rd, "interest_common4385_metrics.json"))
    raw_common = load_json(os.path.join(rd, "raw_common4385_metrics.json"))

    summary = {
        "experiment": "9_sft_with_interest_summary",
        "interest_test4548": pick(interest_4548),
        "raw_test4548": pick(raw_4548),
        "interest_common4385": pick(interest_common),
        "raw_common4385": pick(raw_common),
        "delta_interest_minus_raw_test4548": {
            "HR@1": interest_4548["HR@1"] - raw_4548["HR@1"],
            "HR@10": interest_4548["HR@10"] - raw_4548["HR@10"],
            "NDCG@10": interest_4548["NDCG@10"] - raw_4548["NDCG@10"],
        },
        "delta_interest_minus_raw_common4385": {
            "HR@1": interest_common["HR@1"] - raw_common["HR@1"],
            "HR@10": interest_common["HR@10"] - raw_common["HR@10"],
            "NDCG@10": interest_common["NDCG@10"] - raw_common["NDCG@10"],
        },
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
