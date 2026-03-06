#!/usr/bin/env python3
"""
Prepare LLM-based Recommendation Training Data

This script creates training data for LLM fine-tuning on recommendation tasks.
It converts user-item interaction sequences into LLM-friendly prompt format.

Input:
  - All_Beauty.jsonl: Raw review data with user_id, asin, text, timestamp
  - item_titles_unique.json: Unique condensed titles for each item

Output:
  - train.json: Training samples
  - val.json: Validation samples (second-to-last item)
  - test.json: Test samples (last item)

Data Split (same as vanilla SASRec):
  - Last item: test
  - Second to last: validation
  - Rest: training (with sliding window)

Prompt Format:
  User History:
  1. [Item Title] - Review: [Review text]
  2. [Item Title] - Review: [Review text]
  ...

  Based on the user's interaction history, predict the next product they would be interested in.

  Next Product: [Target Item Title]
"""

import json
import os
import random
from collections import defaultdict
from tqdm import tqdm

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEWS_PATH = os.path.join(SCRIPT_DIR, "../../Data/Amazons/data/All_Beauty.jsonl")
TITLES_PATH = os.path.join(SCRIPT_DIR, "../use_Qwen3-1-7B_to_generate_title/item_titles_unique.json")
OUTPUT_DIR = SCRIPT_DIR

MIN_SEQ_LENGTH = 3      # Minimum sequence length
MIN_ITEM_FREQ = 5       # Filter low-frequency items
MAX_HISTORY_LEN = 20    # Maximum history items in prompt (to avoid too long prompts)
MAX_REVIEW_LEN = 150    # Maximum characters for review text


def load_unique_titles():
    """Load unique item titles."""
    print(f"Loading unique titles from {TITLES_PATH}")
    with open(TITLES_PATH, 'r', encoding='utf-8') as f:
        titles_data = json.load(f)

    # Create lookup: item_id -> condensed_title
    titles = {}
    for item in titles_data:
        item_id = item['item_id']
        title = item['condensed_title']
        titles[item_id] = title

    print(f"  Loaded {len(titles)} unique titles")
    return titles


def load_reviews_and_group():
    """Load reviews and group by user."""
    print(f"\nLoading reviews from {REVIEWS_PATH}")

    user_sequences = defaultdict(list)  # user_id -> [(asin, timestamp, rating, review_text)]
    item_freq = defaultdict(int)

    with open(REVIEWS_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading reviews"):
            try:
                record = json.loads(line.strip())
                user_id = record.get('user_id', '')
                asin = record.get('parent_asin') or record.get('asin', '')
                timestamp = record.get('timestamp', 0)
                rating = record.get('rating', 0)
                review_text = record.get('text', '')

                if user_id and asin:
                    user_sequences[user_id].append((asin, timestamp, rating, review_text))
                    item_freq[asin] += 1
            except (json.JSONDecodeError, KeyError):
                continue

    print(f"  Total reviews: {sum(len(seq) for seq in user_sequences.values())}")
    print(f"  Total users: {len(user_sequences)}")
    print(f"  Total items: {len(item_freq)}")

    return user_sequences, item_freq


def filter_data(user_sequences, item_freq, unique_titles):
    """Filter low-frequency items and items without titles."""
    print(f"\n[Filtering] Removing items with freq < {MIN_ITEM_FREQ} or without titles...")

    valid_items = {item for item, freq in item_freq.items()
                   if freq >= MIN_ITEM_FREQ and item in unique_titles}

    print(f"  Items before filtering: {len(item_freq)}")
    print(f"  Items after filtering: {len(valid_items)}")

    filtered_sequences = {}
    for user_id, seq in user_sequences.items():
        filtered_seq = [(item, ts, rating, review)
                        for item, ts, rating, review in seq
                        if item in valid_items]
        if len(filtered_seq) >= MIN_SEQ_LENGTH:
            filtered_sequences[user_id] = filtered_seq

    print(f"  Users before filtering: {len(user_sequences)}")
    print(f"  Users after filtering: {len(filtered_sequences)}")

    return filtered_sequences


def truncate_review(review_text, max_len=MAX_REVIEW_LEN):
    """Truncate review text to max_len characters."""
    if len(review_text) <= max_len:
        return review_text
    return review_text[:max_len-3] + "..."


def create_prompt(history_items, unique_titles, include_target=False, target_item=None):
    """
    Create LLM-friendly prompt for recommendation.

    Args:
        history_items: List of (item_id, rating, review_text) tuples
        unique_titles: Dict of item_id -> title
        include_target: Whether to include the target in output
        target_item: The target item_id if include_target is True
    """
    # Build history section
    history_lines = []
    for i, (item_id, rating, review_text) in enumerate(history_items, 1):
        title = unique_titles.get(item_id, "Unknown Product")
        truncated_review = truncate_review(review_text)
        # Format: "Title" (Rating: X) - Review: ...
        history_lines.append(f'{i}. "{title}" (Rating: {int(rating)}) - Review: {truncated_review}')

    history_text = "\n".join(history_lines)

    # Build full prompt
    if include_target and target_item:
        target_title = unique_titles.get(target_item, "Unknown Product")
        prompt = f"""User's purchase history:
{history_text}

Based on the user's interaction history, predict the next product they would be most interested in purchasing.

Next product: "{target_title}"<|endoftext|>"""
    else:
        prompt = f"""User's purchase history:
{history_text}

Based on the user's interaction history, predict the next product they would be most interested in purchasing.

Next product:"""

    return prompt


def create_samples(user_sequences, unique_titles):
    """
    Create train/val/test samples following vanilla SASRec split:
    - Test: last item (use all but last as history)
    - Val: second-to-last item (use all but last two as history)
    - Train: sliding window on remaining items
    """
    print("\n[Creating Samples] Splitting into train/val/test...")

    train_samples = []
    val_samples = []
    test_samples = []

    for user_id, seq in tqdm(user_sequences.items(), desc="Processing users"):
        # Sort by timestamp
        sorted_seq = sorted(seq, key=lambda x: x[1])
        items = [(item, rating, review) for item, _, rating, review in sorted_seq]

        if len(items) < MIN_SEQ_LENGTH:
            continue

        # Test: last item
        test_history = items[:-1][-MAX_HISTORY_LEN:]  # Last N items as history
        test_target_item = items[-1][0]
        test_prompt = create_prompt(test_history, unique_titles,
                                    include_target=False, target_item=test_target_item)
        test_target_title = unique_titles.get(test_target_item, "Unknown Product")
        test_samples.append({
            "prompt": test_prompt,
            "target_item_id": test_target_item,
            "target_title": test_target_title,
            "user_id": user_id
        })

        # Val: second-to-last item (if sequence long enough)
        if len(items) >= 3:
            val_history = items[:-2][-MAX_HISTORY_LEN:]
            val_target_item = items[-2][0]
            val_prompt = create_prompt(val_history, unique_titles,
                                       include_target=False, target_item=val_target_item)
            val_target_title = unique_titles.get(val_target_item, "Unknown Product")
            val_samples.append({
                "prompt": val_prompt,
                "target_item_id": val_target_item,
                "target_title": val_target_title,
                "user_id": user_id
            })

        # Train: sliding window (exclude last two items which are val/test)
        train_items = items[:-2]  # Exclude val and test items
        for i in range(1, len(train_items)):
            history = train_items[:i][-MAX_HISTORY_LEN:]
            target_item = train_items[i][0]
            prompt = create_prompt(history, unique_titles,
                                   include_target=False, target_item=target_item)
            target_title = unique_titles.get(target_item, "Unknown Product")
            train_samples.append({
                "prompt": prompt,
                "target_item_id": target_item,
                "target_title": target_title,
                "user_id": user_id
            })

    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples: {len(val_samples)}")
    print(f"  Test samples: {len(test_samples)}")

    return train_samples, val_samples, test_samples


def main():
    print("=" * 60)
    print("LLM Recommendation Data Preparation")
    print("=" * 60)

    # Step 1: Load unique titles
    unique_titles = load_unique_titles()

    # Step 2: Load reviews and group by user
    user_sequences, item_freq = load_reviews_and_group()

    # Step 3: Filter data
    filtered_sequences = filter_data(user_sequences, item_freq, unique_titles)

    # Step 4: Create train/val/test samples
    train_samples, val_samples, test_samples = create_samples(filtered_sequences, unique_titles)

    # Step 5: Shuffle train samples
    random.seed(42)
    random.shuffle(train_samples)

    # Step 6: Save datasets
    print(f"\n[Saving] Writing to {OUTPUT_DIR}")

    with open(os.path.join(OUTPUT_DIR, "train.json"), 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, "val.json"), 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, "test.json"), 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, indent=2, ensure_ascii=False)

    # Step 7: Save statistics
    stats = {
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_test": len(test_samples),
        "num_users": len(filtered_sequences),
        "num_items": len(set(s["target_item_id"] for s in train_samples + val_samples + test_samples)),
        "min_seq_length": MIN_SEQ_LENGTH,
        "min_item_freq": MIN_ITEM_FREQ,
        "max_history_len": MAX_HISTORY_LEN,
        "max_review_len": MAX_REVIEW_LEN
    }

    with open(os.path.join(OUTPUT_DIR, "stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Train samples: {stats['num_train']}")
    print(f"Val samples: {stats['num_val']}")
    print(f"Test samples: {stats['num_test']}")
    print(f"Unique users: {stats['num_users']}")
    print(f"Unique items: {stats['num_items']}")

    # Print sample prompts
    print("\n" + "=" * 60)
    print("Sample Training Prompt")
    print("=" * 60)
    if train_samples:
        print(train_samples[0]["prompt"])
        print(f"\n[Expected output: \"{train_samples[0]['target_title']}\"]")

    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"\nOutput files in {OUTPUT_DIR}:")
    print("  - train.json")
    print("  - val.json")
    print("  - test.json")
    print("  - stats.json")


if __name__ == "__main__":
    main()
