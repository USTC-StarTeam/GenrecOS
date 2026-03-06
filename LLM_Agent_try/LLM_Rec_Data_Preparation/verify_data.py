#!/usr/bin/env python3
"""
Verify the generated LLM recommendation data.

This script checks:
1. Data integrity (all fields present)
2. Title consistency (target_title matches target_item_id in titles lookup)
3. Distribution statistics
4. Sample inspection
"""

import json
import os
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TITLES_PATH = os.path.join(SCRIPT_DIR, "../use_Qwen3-1-7B_to_generate_title/item_titles_unique.json")


def load_titles():
    """Load unique titles lookup."""
    with open(TITLES_PATH, 'r', encoding='utf-8') as f:
        titles_data = json.load(f)
    return {item['item_id']: item['condensed_title'] for item in titles_data}


def verify_split(split_name, data, titles_lookup):
    """Verify a data split."""
    print(f"\n{'='*60}")
    print(f"Verifying {split_name}")
    print('='*60)

    errors = []
    history_lengths = []
    review_lengths = []
    target_titles = []

    for i, sample in enumerate(data):
        # Check required fields
        required_fields = ['prompt', 'target_item_id', 'target_title', 'user_id']
        for field in required_fields:
            if field not in sample:
                errors.append(f"Sample {i}: Missing field '{field}'")

        # Check title consistency
        if sample['target_item_id'] in titles_lookup:
            expected_title = titles_lookup[sample['target_item_id']]
            if sample['target_title'] != expected_title:
                errors.append(f"Sample {i}: Title mismatch for {sample['target_item_id']}")

        # Collect statistics
        history_lengths.append(sample['prompt'].count('\n'))
        target_titles.append(sample['target_title'])

    # Print statistics
    print(f"Total samples: {len(data)}")
    print(f"Unique users: {len(set(s['user_id'] for s in data))}")
    print(f"Unique target items: {len(set(s['target_item_id'] for s in data))}")
    print(f"History lines - Min: {min(history_lengths)}, Max: {max(history_lengths)}, Avg: {sum(history_lengths)/len(history_lengths):.1f}")

    # Most common target titles
    title_counter = Counter(target_titles)
    print(f"\nTop 10 most common target titles:")
    for title, count in title_counter.most_common(10):
        print(f"  '{title}': {count}")

    if errors:
        print(f"\n⚠️  Found {len(errors)} errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print("\n✅ No errors found!")

    return len(errors) == 0


def main():
    print("="*60)
    print("LLM Recommendation Data Verification")
    print("="*60)

    # Load titles lookup
    print("\nLoading titles lookup...")
    titles_lookup = load_titles()
    print(f"  Loaded {len(titles_lookup)} titles")

    # Load and verify each split
    all_valid = True

    for split in ['train', 'val', 'test']:
        path = os.path.join(SCRIPT_DIR, f"{split}.json")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        valid = verify_split(split, data, titles_lookup)
        all_valid = all_valid and valid

    # Overall summary
    print("\n" + "="*60)
    print("Overall Summary")
    print("="*60)

    if all_valid:
        print("✅ All data splits are valid!")
    else:
        print("❌ Some data splits have errors. Please check above.")

    # Print sample prompts for manual inspection
    print("\n" + "="*60)
    print("Sample Prompt (first train sample)")
    print("="*60)
    with open(os.path.join(SCRIPT_DIR, "train.json"), 'r') as f:
        train_data = json.load(f)
    print(train_data[0]['prompt'])
    print(f"\nExpected output: \"{train_data[0]['target_title']}\"")


if __name__ == "__main__":
    main()
