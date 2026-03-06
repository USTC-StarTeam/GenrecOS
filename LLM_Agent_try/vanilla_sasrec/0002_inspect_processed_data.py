#!/usr/bin/env python3
"""
Inspect the processed data to verify correctness
"""

import json
import os

processed_data_dir = "./processed_data"

print("=" * 60)
print("Inspecting Processed Beauty Data")
print("=" * 60)

# Check item mapping
item_mapping_path = os.path.join(processed_data_dir, "item_mapping.json")
if os.path.exists(item_mapping_path):
    with open(item_mapping_path, 'r') as f:
        item_mapping = json.load(f)

    print("\n[Item Mapping]")
    print(f"  Total items: {item_mapping['num_items']}")
    print(f"  Sample mappings (first 5):")
    for i, (item, idx) in enumerate(list(item_mapping['item_to_id'].items())[:5]):
        print(f"    {item} -> {idx}")
else:
    print(f"\n[ERROR] Item mapping not found at {item_mapping_path}")
    exit(1)

# Check splits
splits_dir = os.path.join(processed_data_dir, "splits")
if os.path.exists(splits_dir):
    for split in ["train", "validation", "test"]:
        split_path = os.path.join(splits_dir, f"{split}.json")
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                data = json.load(f)

            print(f"\n[{split.upper()} Split]")
            print(f"  Total samples: {len(data)}")

            if len(data) > 0:
                # Sample structure
                sample = data[0]
                print(f"  Sample keys: {list(sample.keys())}")
                print(f"  Sample prompt (first 100 chars): {sample['prompt'][:100]}...")
                print(f"  Sample ground_truth: {sample['ground_truth']}")

                # Sequence length stats
                seq_lens = [len(s['prompt'].split()) for s in data]
                print(f"  Sequence length stats:")
                print(f"    Min: {min(seq_lens)}")
                print(f"    Max: {max(seq_lens)}")
                print(f"    Mean: {sum(seq_lens) / len(seq_lens):.2f}")

                # Show a few samples
                print(f"\n  Sample data points:")
                for i in range(min(3, len(data))):
                    s = data[i]
                    prompt_tokens = s['prompt'].split()
                    # Show last 5 tokens of prompt
                    if len(prompt_tokens) > 5:
                        preview = " ... " + " ".join(prompt_tokens[-5:])
                    else:
                        preview = " " + s['prompt']
                    print(f"    [{i}] prompt:{preview}")
                    print(f"        ground_truth: {s['ground_truth']}")
else:
    print(f"\n[ERROR] Splits directory not found at {splits_dir}")
    exit(1)

print("\n" + "=" * 60)
print("Data Inspection Complete!")
print("=" * 60)
