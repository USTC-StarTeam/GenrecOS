#!/usr/bin/env python3

import argparse
import os
from collections import defaultdict

import yaml

from pipeline_utils import (
    PatternMatcher,
    build_item_integer_mapping,
    build_mining_sequences,
    build_split_samples,
    dump_json,
    dump_jsonl,
    ensure_dir,
    load_item_titles,
    mine_patterns_with_seq2pat,
    parse_raw_sequences,
    sample_to_row,
    summarize_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SFT data with seq2pat memory-enhanced prompts.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sft_pattern_config.yaml"),
        help="Path to config yaml.",
    )
    parser.add_argument(
        "--max_users",
        type=int,
        default=0,
        help="Optional smoke mode: keep at most this many users after filtering.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths_cfg = config["paths"]
    data_cfg = config["data"]
    pattern_cfg = config["pattern_memory"]

    data_dir = os.path.join(script_dir, paths_cfg["data_dir"])
    patterns_dir = os.path.join(script_dir, paths_cfg["patterns_dir"])
    ensure_dir(data_dir)
    ensure_dir(patterns_dir)

    raw_data_path = os.path.join(script_dir, paths_cfg["raw_data_path"])
    titles_path = os.path.join(script_dir, paths_cfg["item_titles_path"])

    item_to_title, norm_title_to_items = load_item_titles(titles_path)
    user_sequences = parse_raw_sequences(
        raw_data_path=raw_data_path,
        item_to_title=item_to_title,
        min_item_freq=int(data_cfg["min_item_freq"]),
        min_seq_length=int(data_cfg["min_seq_length"]),
    )

    if args.max_users > 0:
        user_ids = sorted(user_sequences.keys())[: args.max_users]
        user_sequences = {u: user_sequences[u] for u in user_ids}

    item_to_int, int_to_item = build_item_integer_mapping(user_sequences)
    mining_sequences = build_mining_sequences(
        user_sequences=user_sequences,
        item_to_int=item_to_int,
        trim_last=int(pattern_cfg["trim_last_for_mining"]),
        min_len=int(pattern_cfg["min_pattern_len"]),
    )

    patterns = mine_patterns_with_seq2pat(
        mining_sequences=mining_sequences,
        int_to_item=int_to_item,
        item_to_title=item_to_title,
        min_frequency=int(pattern_cfg["min_frequency"]),
        min_pattern_len=int(pattern_cfg["min_pattern_len"]),
        max_pattern_len=int(pattern_cfg["max_pattern_len"]),
        max_patterns=int(pattern_cfg["max_patterns"]),
        max_span=int(pattern_cfg["max_span"]),
        n_jobs=int(pattern_cfg["n_jobs"]),
        seed=int(pattern_cfg["seed"]),
    )
    dump_json(os.path.join(patterns_dir, "seq2pat_patterns.json"), patterns)
    dump_json(
        os.path.join(patterns_dir, "seq2pat_patterns_preview.json"),
        {
            "num_patterns": len(patterns),
            "top_patterns": patterns[:100],
        },
    )

    matcher = PatternMatcher(
        patterns=patterns,
        partial_min_ratio=float(pattern_cfg["partial_min_ratio"]),
        partial_min_matched=int(pattern_cfg["partial_min_matched"]),
        max_matches=int(pattern_cfg["max_matches_per_sample"]),
    )

    samples = build_split_samples(
        user_sequences=user_sequences,
        item_to_title=item_to_title,
        max_history=int(data_cfg["max_history"]),
        max_review_len=int(data_cfg["max_review_len"]),
        matcher=matcher,
        max_patterns_in_prompt=int(pattern_cfg["max_patterns_in_prompt"]),
    )

    rows_by_split = defaultdict(list)
    for sample in samples:
        rows_by_split[sample.split].append(sample_to_row(sample, int(pattern_cfg["max_patterns_in_prompt"])))

    for split in ("train", "val", "test"):
        dump_jsonl(os.path.join(data_dir, f"{split}.jsonl"), rows_by_split[split])

    summary = {
        "task": "seq2pat_memory_prompt_sft",
        "config": {
            "data": data_cfg,
            "pattern_memory": pattern_cfg,
        },
        "num_unique_condensed_titles": len(norm_title_to_items),
        "num_unique_items": len(item_to_title),
        "num_users_after_filtering": len(user_sequences),
        "num_item_ids_in_memory_mapping": len(item_to_int),
        "mining_sequence_count": len(mining_sequences),
        "mined_pattern_count": len(patterns),
        "train": summarize_rows(rows_by_split["train"]),
        "val": summarize_rows(rows_by_split["val"]),
        "test": summarize_rows(rows_by_split["test"]),
        "example_train_row": rows_by_split["train"][0] if rows_by_split["train"] else None,
    }
    dump_json(os.path.join(data_dir, "dataset_summary.json"), summary)

    print("Prepared seq2pat-memory SFT datasets:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {len(rows_by_split[split])}")
    print(f"Patterns: {len(patterns)}")
    print(f"Summary written to {os.path.join(data_dir, 'dataset_summary.json')}")


if __name__ == "__main__":
    main()
