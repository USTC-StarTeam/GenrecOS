#!/usr/bin/env python3

import argparse
import os
from collections import defaultdict

import yaml

from pipeline_utils import (
    build_item_review_pool,
    build_split_samples,
    dump_json,
    dump_jsonl,
    ensure_dir,
    load_item_titles,
    parse_filtered_sequences,
    sample_to_row,
    set_seed,
    summarize_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare review-retrieval augmented SFT data.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "experiment_config.yaml"),
        help="Path to config yaml.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(int(config["training"]["seed"]))
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    retrieval_cfg = config["retrieval"]

    data_dir = os.path.join(script_dir, paths_cfg["data_dir"])
    ensure_dir(data_dir)

    raw_data_path = os.path.join(script_dir, paths_cfg["raw_data_path"])
    titles_path = os.path.join(script_dir, paths_cfg["item_titles_path"])
    item_to_title, norm_title_to_items = load_item_titles(titles_path)

    user_sequences = parse_filtered_sequences(
        raw_data_path=raw_data_path,
        item_to_title=item_to_title,
        min_item_freq=data_cfg["min_item_freq"],
        min_seq_length=data_cfg["min_seq_length"],
    )
    item_review_pool = build_item_review_pool(
        user_sequences=user_sequences,
        item_to_title=item_to_title,
        retrieved_review_max_len=retrieval_cfg["retrieved_review_max_len"],
        pool_size_per_item=retrieval_cfg["pool_size_per_item"],
        min_review_chars=int(retrieval_cfg.get("min_review_chars", 0)),
        min_review_quality=float(retrieval_cfg.get("min_review_quality", 0.0)),
    )

    samples = build_split_samples(
        user_sequences=user_sequences,
        item_to_title=item_to_title,
        item_review_pool=item_review_pool,
        max_history=data_cfg["max_history"],
        max_review_len=data_cfg["max_review_len"],
        short_history_threshold=retrieval_cfg["short_history_threshold"],
        max_aug_reviews_per_sample=retrieval_cfg["max_aug_reviews_per_sample"],
        max_aug_reviews_per_item=retrieval_cfg["max_aug_reviews_per_item"],
        recent_first=bool(retrieval_cfg.get("recent_first", True)),
        recent_item_window=retrieval_cfg.get("recent_item_window"),
        require_same_rating_bucket=bool(retrieval_cfg.get("require_same_rating_bucket", False)),
    )

    rows_by_split = defaultdict(list)
    for sample in samples:
        rows_by_split[sample.split].append(sample_to_row(sample))

    for split in ("train", "val", "test"):
        dump_jsonl(os.path.join(data_dir, f"{split}.jsonl"), rows_by_split[split])

    short_threshold = retrieval_cfg["short_history_threshold"]
    summary = {
        "task": "retrieval_augmented_next_item_sft",
        "config": {
            "data": data_cfg,
            "retrieval": retrieval_cfg,
        },
        "num_unique_condensed_titles": len(norm_title_to_items),
        "num_unique_items": len(item_to_title),
        "num_users_after_filtering": len(user_sequences),
        "num_item_review_pools": len(item_review_pool),
        "full_sequence_short_user_count": sum(1 for seq in user_sequences.values() if len(seq) <= short_threshold),
        "sample_level_short_history_count": {
            split: sum(1 for row in rows_by_split[split] if len(row["history_titles"]) <= short_threshold)
            for split in ("train", "val", "test")
        },
        "train": summarize_rows(rows_by_split["train"]),
        "val": summarize_rows(rows_by_split["val"]),
        "test": summarize_rows(rows_by_split["test"]),
        "example": rows_by_split["train"][0] if rows_by_split["train"] else None,
    }
    dump_json(os.path.join(data_dir, "dataset_summary.json"), summary)

    print("Prepared augmented SFT datasets:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {len(rows_by_split[split])}")
    print(f"Summary written to {os.path.join(data_dir, 'dataset_summary.json')}")


if __name__ == "__main__":
    main()
