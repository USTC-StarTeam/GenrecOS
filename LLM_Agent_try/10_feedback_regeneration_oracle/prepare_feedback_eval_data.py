#!/usr/bin/env python3

import argparse
import os
from typing import Dict, List

import yaml

from pipeline_utils import dump_json, dump_jsonl, ensure_dir, load_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare eval datasets for feedback-regeneration experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "feedback_regen_config.yaml"),
        help="Path to config yaml.",
    )
    return parser.parse_args()


def validate_rows(rows: List[dict], source_path: str) -> None:
    required = {"user_id", "target_item_id", "target_title", "prompt"}
    for idx, row in enumerate(rows):
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"{source_path}: row {idx} missing keys: {missing}")


def prepare_dataset(
    rows: List[dict],
    dataset_name: str,
    trigger_top_k: int,
    max_feedback_rounds: int,
    feedback_message: str,
) -> List[dict]:
    prepared_rows: List[dict] = []
    for row in rows:
        prepared = dict(row)
        prepared["feedback_dataset"] = dataset_name
        prepared["feedback_trigger_top_k"] = trigger_top_k
        prepared["max_feedback_rounds"] = max_feedback_rounds
        prepared["feedback_message"] = feedback_message
        prepared_rows.append(prepared)
    return prepared_rows


def summarize_dataset(rows: List[dict]) -> Dict[str, float]:
    history_lengths = [len(row.get("history_titles", [])) for row in rows]
    prompt_lens = [len(row["prompt"]) for row in rows]
    return {
        "samples": len(rows),
        "unique_users": len({row["user_id"] for row in rows}),
        "unique_targets": len({row["target_item_id"] for row in rows}),
        "history_len_avg": round(sum(history_lengths) / len(history_lengths), 2) if history_lengths else 0.0,
        "prompt_chars_avg": round(sum(prompt_lens) / len(prompt_lens), 2) if prompt_lens else 0.0,
    }


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths_cfg = config["paths"]
    eval_cfg = config["evaluation"]
    feedback_cfg = config["feedback"]

    data_dir = os.path.join(script_dir, paths_cfg["data_dir"])
    ensure_dir(data_dir)

    trigger_top_k = int(feedback_cfg["trigger_top_k"])
    max_feedback_rounds = int(feedback_cfg["max_feedback_rounds"])
    feedback_message = str(feedback_cfg["feedback_message"])

    summary = {
        "experiment": "10_feedback_regeneration_oracle",
        "trigger_top_k": trigger_top_k,
        "max_feedback_rounds": max_feedback_rounds,
        "datasets": {},
    }

    for ds in eval_cfg["datasets"]:
        source_key = ds["source_key"]
        source_rel = paths_cfg[source_key]
        source_path = os.path.join(script_dir, source_rel)
        output_path = os.path.join(data_dir, ds["prepared_file"])

        rows = load_jsonl(source_path)
        validate_rows(rows, source_path)
        prepared_rows = prepare_dataset(
            rows=rows,
            dataset_name=ds["name"],
            trigger_top_k=trigger_top_k,
            max_feedback_rounds=max_feedback_rounds,
            feedback_message=feedback_message,
        )
        dump_jsonl(output_path, prepared_rows)

        summary["datasets"][ds["name"]] = {
            "source_path": source_path,
            "prepared_path": output_path,
            **summarize_dataset(prepared_rows),
        }
        print(f"[prepare] {ds['name']}: {len(prepared_rows)} rows -> {output_path}")

    summary_path = os.path.join(data_dir, "feedback_dataset_summary.json")
    dump_json(summary_path, summary)
    print(f"[prepare] summary -> {summary_path}")


if __name__ == "__main__":
    main()
