#!/usr/bin/env python3
"""
Analyze the Beauty dataset to understand sequence lengths
"""

import json
import os
from collections import defaultdict
from datetime import datetime

# Data path
data_path = "../../Data/Amazons/data/All_Beauty.jsonl"

print(f"Analyzing {data_path}...")
print("=" * 60)

# Group interactions by user_id
user_sequences = defaultdict(list)

total_lines = 0
with open(data_path, 'r') as f:
    for line in f:
        total_lines += 1
        if total_lines % 100000 == 0:
            print(f"Processed {total_lines} lines...")

        try:
            record = json.loads(line.strip())
            user_id = record.get('user_id', '')
            asin = record.get('asin', '')
            timestamp = record.get('timestamp', 0)

            if user_id and asin:
                user_sequences[user_id].append((asin, timestamp))
        except json.JSONDecodeError:
            continue

print(f"\nTotal interactions: {total_lines}")
print(f"Total unique users: {len(user_sequences)}")

# Sort sequences by timestamp and compute statistics
seq_lengths = []
for user_id in user_sequences:
    seq = sorted(user_sequences[user_id], key=lambda x: x[1])
    seq_lengths.append(len(seq))

# Basic statistics
import numpy as np
seq_lengths = np.array(seq_lengths)

print("\n" + "=" * 60)
print("Sequence Length Statistics:")
print("=" * 60)
print(f"Min sequence length:     {seq_lengths.min()}")
print(f"Max sequence length:     {seq_lengths.max()}")
print(f"Mean sequence length:    {seq_lengths.mean():.2f}")
print(f"Median sequence length:  {np.median(seq_lengths):.2f}")
print(f"Std sequence length:     {seq_lengths.std():.2f}")

# Percentiles
print("\nSequence Length Percentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}%th percentile: {np.percentile(seq_lengths, p):.0f}")

# Filter users by minimum sequence length
print("\n" + "=" * 60)
print("Users by Minimum Sequence Length:")
print("=" * 60)
for min_len in [3, 5, 10, 20, 50]:
    count = np.sum(seq_lengths >= min_len)
    pct = count / len(seq_lengths) * 100
    print(f"Users with >= {min_len:2d} interactions: {count:6d} ({pct:5.2f}%)")

# Unique items
all_items = set()
for user_id in user_sequences:
    for asin, _ in user_sequences[user_id]:
        all_items.add(asin)

print(f"\nTotal unique items: {len(all_items)}")

# Item frequency analysis
item_freq = defaultdict(int)
for user_id in user_sequences:
    for asin, _ in user_sequences[user_id]:
        item_freq[asin] += 1

item_freq_list = list(item_freq.values())
item_freq_arr = np.array(item_freq_list)

print("\n" + "=" * 60)
print("Item Frequency Statistics:")
print("=" * 60)
print(f"Min item frequency:     {item_freq_arr.min()}")
print(f"Max item frequency:     {item_freq_arr.max()}")
print(f"Mean item frequency:    {item_freq_arr.mean():.2f}")
print(f"Median item frequency:  {np.median(item_freq_arr):.2f}")

# Items with frequency >= k
print("\nItems by Minimum Frequency:")
for k in [5, 10, 20, 50]:
    count = np.sum(item_freq_arr >= k)
    print(f"Items with >= {k:2d} interactions: {count:6d}")

# Time span analysis
print("\n" + "=" * 60)
print("Time Span Analysis:")
print("=" * 60)

timestamps = []
for user_id in user_sequences:
    for _, ts in user_sequences[user_id]:
        timestamps.append(ts)

timestamps = sorted(timestamps)
if timestamps:
    print(f"Earliest timestamp: {datetime.fromtimestamp(timestamps[0]/1000).isoformat()}")
    print(f"Latest timestamp:   {datetime.fromtimestamp(timestamps[-1]/1000).isoformat()}")
    print(f"Time span:          {(timestamps[-1] - timestamps[0]) / (1000 * 60 * 60 * 24):.0f} days")

# Split analysis (based on sequence position)
print("\n" + "=" * 60)
print("Data Split Analysis (chronological):")
print("=" * 60)

train_count = 0
val_count = 0
test_count = 0
min_seq_len = 3

for user_id in user_sequences:
    seq = sorted(user_sequences[user_id], key=lambda x: x[1])
    if len(seq) >= min_seq_len:
        train_count += len(seq) - 2  # All except last 2
        val_count += 1  # Second to last
        test_count += 1  # Last

print(f"Training interactions:   {train_count}")
print(f"Validation samples:      {val_count}")
print(f"Test samples:            {test_count}")
print(f"Total users (>=3 items): {sum(1 for s in seq_lengths if s >= min_len)}")

# Recommended max_seq_length
print("\n" + "=" * 60)
print("Recommendations:")
print("=" * 60)

p95_len = int(np.percentile(seq_lengths, 95))
p90_len = int(np.percentile(seq_lengths, 90))
p75_len = int(np.percentile(seq_lengths, 75))

print(f"For 95% coverage: max_seq_length = {p95_len}")
print(f"For 90% coverage: max_seq_length = {p90_len}")
print(f"For 75% coverage: max_seq_length = {p75_len}")
print(f"\nRecommended max_seq_length: {min(100, p90_len)} (for shorter sequences)")

# Save summary
summary = {
    "total_interactions": int(total_lines),
    "total_users": len(user_sequences),
    "total_items": len(all_items),
    "seq_length_stats": {
        "min": int(seq_lengths.min()),
        "max": int(seq_lengths.max()),
        "mean": float(seq_lengths.mean()),
        "median": float(np.median(seq_lengths)),
        "std": float(seq_lengths.std()),
        "p90": int(np.percentile(seq_lengths, 90)),
        "p95": int(np.percentile(seq_lengths, 95)),
    },
    "item_freq_stats": {
        "min": int(item_freq_arr.min()),
        "max": int(item_freq_arr.max()),
        "mean": float(item_freq_arr.mean()),
        "median": float(np.median(item_freq_arr)),
    },
    "recommended_max_seq_length": min(100, p90_len),
}

with open("data_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to data_summary.json")
