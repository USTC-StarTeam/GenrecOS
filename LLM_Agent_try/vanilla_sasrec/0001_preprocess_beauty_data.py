#!/usr/bin/env python3
"""
Preprocess Beauty dataset for SASRec training
Converts All_Beauty.jsonl to train/val/test JSON format

Data split strategy (chronological):
- Last item: test
- Second to last: validation
- Rest: training

Each sample format:
{
    "prompt": "<item_id_1> <item_id_2> ... <item_id_n>",
    "ground_truth": "<target_item_id>"
}
"""

import json
import os
import random
from collections import defaultdict
from tqdm import tqdm

# Config
INPUT_PATH = "../../Data/Amazons/data/All_Beauty.jsonl"
OUTPUT_DIR = "./processed_data"
MIN_SEQ_LENGTH = 3  # Minimum sequence length to include
MIN_ITEM_FREQ = 5   # Filter items with low frequency
TRAIN_SPLIT_NEG_SAMPLES = 0  # Negative samples for training (0 = only positive)

print("=" * 60)
print("Beauty Dataset Preprocessing for SASRec")
print("=" * 60)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load and group interactions by user
print("\n[Step 1] Loading data and grouping by user...")
user_sequences = defaultdict(list)
item_freq = defaultdict(int)

with open(INPUT_PATH, 'r') as f:
    for line in tqdm(f, desc="Reading interactions"):
        try:
            record = json.loads(line.strip())
            user_id = record.get('user_id', '')
            asin = record.get('asin', '')
            timestamp = record.get('timestamp', 0)

            if user_id and asin:
                user_sequences[user_id].append((asin, timestamp))
                item_freq[asin] += 1
        except (json.JSONDecodeError, KeyError):
            continue

print(f"  Total interactions: {sum(len(seq) for seq in user_sequences.values())}")
print(f"  Total users: {len(user_sequences)}")
print(f"  Total items: {len(item_freq)}")

# Step 2: Filter low-frequency items
print(f"\n[Step 2] Filtering items with frequency < {MIN_ITEM_FREQ}...")
valid_items = {item for item, freq in item_freq.items() if freq >= MIN_ITEM_FREQ}
print(f"  Items before filtering: {len(item_freq)}")
print(f"  Items after filtering: {len(valid_items)}")

# Filter sequences
filtered_user_sequences = {}
for user_id, seq in user_sequences.items():
    # Keep only interactions with valid items
    filtered_seq = [(item, ts) for item, ts in seq if item in valid_items]
    if len(filtered_seq) >= MIN_SEQ_LENGTH:
        filtered_user_sequences[user_id] = filtered_seq

print(f"  Users before filtering: {len(user_sequences)}")
print(f"  Users after filtering: {len(filtered_user_sequences)}")

# Step 3: Build item ID mapping (remap to 0, 1, 2, ...)
print("\n[Step 3] Building item ID mapping...")
sorted_items = sorted(valid_items)
item_to_id = {item: idx for idx, item in enumerate(sorted_items)}
id_to_item = {idx: item for item, idx in item_to_id.items()}

print(f"  Item ID range: 0 to {len(item_to_id) - 1}")

# Save item mapping
with open(os.path.join(OUTPUT_DIR, "item_mapping.json"), 'w') as f:
    json.dump({
        "item_to_id": item_to_id,
        "id_to_item": id_to_item,
        "num_items": len(item_to_id)
    }, f, indent=2)
print(f"  Saved item mapping to {OUTPUT_DIR}/item_mapping.json")

# Step 4: Sort sequences by timestamp and create train/val/test samples
print("\n[Step 4] Creating train/val/test samples...")

train_samples = []
val_samples = []
test_samples = []

for user_id, seq in tqdm(filtered_user_sequences.items(), desc="Processing users"):
    # Sort by timestamp
    sorted_seq = sorted(seq, key=lambda x: x[1])
    item_sequence = [item_to_id[item] for item, _ in sorted_seq]

    # Skip if sequence too short after filtering
    if len(item_sequence) < MIN_SEQ_LENGTH:
        continue

    # Test: last item
    test_prompt = " ".join(map(str, item_sequence[:-1]))
    test_target = str(item_sequence[-1])
    test_samples.append({
        "prompt": test_prompt,
        "ground_truth": test_target,
        "user_id": user_id
    })

    # Val: second to last item (use sequence up to third-to-last as input)
    if len(item_sequence) >= 3:
        val_prompt = " ".join(map(str, item_sequence[:-2]))
        val_target = str(item_sequence[-2])
        val_samples.append({
            "prompt": val_prompt,
            "ground_truth": val_target,
            "user_id": user_id
        })

    # Train: all other items (create sliding window sequences)
    # For training, we create sequences of varying lengths
    # Each position predicts the next item
    for i in range(1, len(item_sequence) - 1):  # Exclude val and test positions
        train_prompt = " ".join(map(str, item_sequence[:i]))
        train_target = str(item_sequence[i])
        train_samples.append({
            "prompt": train_prompt,
            "ground_truth": train_target,
            "user_id": user_id
        })

print(f"  Train samples: {len(train_samples)}")
print(f"  Val samples: {len(val_samples)}")
print(f"  Test samples: {len(test_samples)}")

# Shuffle train samples
random.seed(42)
random.shuffle(train_samples)

# Step 5: Save datasets
print("\n[Step 5] Saving datasets...")

with open(os.path.join(OUTPUT_DIR, "train.json"), 'w') as f:
    json.dump(train_samples, f)

with open(os.path.join(OUTPUT_DIR, "val.json"), 'w') as f:
    json.dump(val_samples, f)

with open(os.path.join(OUTPUT_DIR, "test.json"), 'w') as f:
    json.dump(test_samples, f)

# Also save in format expected by load_dataset (as directory with split files)
import shutil
split_dir = os.path.join(OUTPUT_DIR, "splits")
os.makedirs(split_dir, exist_ok=True)

with open(os.path.join(split_dir, "train.json"), 'w') as f:
    json.dump(train_samples, f)

with open(os.path.join(split_dir, "validation.json"), 'w') as f:
    json.dump(val_samples, f)

with open(os.path.join(split_dir, "test.json"), 'w') as f:
    json.dump(test_samples, f)

print(f"  Saved to {OUTPUT_DIR}/")
print(f"  Saved to {OUTPUT_DIR}/splits/")

# Step 6: Print statistics
print("\n" + "=" * 60)
print("Dataset Statistics")
print("=" * 60)

# Train sequence lengths
train_seq_lens = [len(s["prompt"].split()) for s in train_samples]
print(f"\nTrain sequence lengths:")
print(f"  Min: {min(train_seq_lens)}")
print(f"  Max: {max(train_seq_lens)}")
print(f"  Mean: {sum(train_seq_lens) / len(train_seq_lens):.2f}")

# Count users in splits
train_users = set(s["user_id"] for s in train_samples)
val_users = set(s["user_id"] for s in val_samples)
test_users = set(s["user_id"] for s in test_samples)

print(f"\nUnique users:")
print(f"  Train: {len(train_users)}")
print(f"  Val: {len(val_users)}")
print(f"  Test: {len(test_users)}")

print("\n" + "=" * 60)
print("Preprocessing Complete!")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR}/")
print(f"  - train.json ({len(train_samples)} samples)")
print(f"  - val.json ({len(val_samples)} samples)")
print(f"  - test.json ({len(test_samples)} samples)")
print(f"  - item_mapping.json ({len(item_to_id)} items)")
print(f"  - splits/ (for load_dataset)")
