#!/usr/bin/env python3

import argparse
import json
import os
from typing import Dict, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build common 4385 aligned test subset for 7_ experiments.")
    parser.add_argument("--input_test_path", type=str, required=True)
    parser.add_argument("--output_test_path", type=str, required=True)
    parser.add_argument(
        "--sasrec_test_path",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "vanilla_sasrec",
            "processed_data",
            "test.json",
        ),
    )
    parser.add_argument(
        "--item_mapping_path",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "vanilla_sasrec",
            "processed_data",
            "item_mapping.json",
        ),
    )
    parser.add_argument("--summary_path", type=str, default=None)
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    sasrec_test = json.load(open(args.sasrec_test_path, "r", encoding="utf-8"))
    mapping = json.load(open(args.item_mapping_path, "r", encoding="utf-8"))
    item_to_id = mapping["item_to_id"]

    sasrec_pairs: Set[Tuple[str, int]] = set()
    for row in sasrec_test:
        sasrec_pairs.add((row["user_id"], int(row["ground_truth"])))

    src_rows = load_jsonl(args.input_test_path)
    kept_rows = []
    dropped_unmapped = 0
    dropped_unaligned = 0

    for row in src_rows:
        user_id = row["user_id"]
        target_item_id = row["target_item_id"]
        internal_id = item_to_id.get(target_item_id)
        if internal_id is None:
            dropped_unmapped += 1
            continue
        if (user_id, int(internal_id)) not in sasrec_pairs:
            dropped_unaligned += 1
            continue
        kept_rows.append(row)

    dump_jsonl(args.output_test_path, kept_rows)

    summary = {
        "input_test_path": args.input_test_path,
        "output_test_path": args.output_test_path,
        "input_size": len(src_rows),
        "output_size": len(kept_rows),
        "dropped_unmapped": dropped_unmapped,
        "dropped_unaligned": dropped_unaligned,
        "sasrec_test_size": len(sasrec_test),
    }

    summary_path = args.summary_path
    if summary_path is None:
        summary_path = args.output_test_path + ".summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
