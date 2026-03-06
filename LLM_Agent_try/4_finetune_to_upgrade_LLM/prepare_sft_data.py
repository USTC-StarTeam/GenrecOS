#!/usr/bin/env python3

import argparse
import os
from collections import defaultdict

import yaml

from pipeline_utils import (
    build_split_samples,
    dump_json,
    dump_jsonl,
    ensure_dir,
    load_item_titles,
    parse_raw_sequences,
    sample_to_row,
    summarize_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare title-only SFT data for Qwen.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sft_config.yaml"),
        help="Path to config yaml.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths_cfg = config["paths"]
    data_cfg = config["data"]

    data_dir = os.path.join(script_dir, paths_cfg["data_dir"])
    ensure_dir(data_dir)

    raw_data_path = os.path.join(script_dir, paths_cfg["raw_data_path"])
    titles_path = os.path.join(script_dir, paths_cfg["item_titles_path"])

    item_to_title, norm_title_to_items = load_item_titles(titles_path)
    user_sequences = parse_raw_sequences(
        raw_data_path=raw_data_path,
        item_to_title=item_to_title,
        min_item_freq=data_cfg["min_item_freq"],
        min_seq_length=data_cfg["min_seq_length"],
    )
    samples = build_split_samples(
        user_sequences=user_sequences,
        item_to_title=item_to_title,
        max_history=data_cfg["max_history"],
        max_review_len=data_cfg["max_review_len"],
    )

    rows_by_split = defaultdict(list)
    for sample in samples:
        rows_by_split[sample.split].append(sample_to_row(sample))

    for split in ("train", "val", "test"):
        output_path = os.path.join(data_dir, f"{split}.jsonl")
        dump_jsonl(output_path, rows_by_split[split])

    summary = {
        "task": "title_only_next_item_sft",
        "config": data_cfg,
        "num_unique_condensed_titles": len(norm_title_to_items),
        "num_unique_items": len(item_to_title),
        "num_users_after_filtering": len(user_sequences),
        "train": summarize_rows(rows_by_split["train"]),
        "val": summarize_rows(rows_by_split["val"]),
        "test": summarize_rows(rows_by_split["test"]),
        "example": rows_by_split["train"][0] if rows_by_split["train"] else None,
    }
    dump_json(os.path.join(data_dir, "dataset_summary.json"), summary)

    print("Prepared SFT datasets:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {len(rows_by_split[split])}")
    print(f"Summary written to {os.path.join(data_dir, 'dataset_summary.json')}")


if __name__ == "__main__":
    main()
