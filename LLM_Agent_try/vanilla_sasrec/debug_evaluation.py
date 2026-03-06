#!/usr/bin/env python3
"""
Debug script to analyze why evaluation metrics are so low
"""

import sys
import os
import json
import torch
import numpy as np
from collections import Counter

sys.path.append("../../Rec-Transformer")
sys.path.append("../..")

from sasrec import SasRecForCausalLM
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader

print("=" * 60)
print("Debug: Analyzing SASRec Predictions")
print("=" * 60)

# Load model and tokenizer
model_path = "./checkpoints/sasrec_beauty_20260226_054647/best_model"
print(f"\nLoading model from {model_path}...")
model = SasRecForCausalLM.from_pretrained(model_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"PAD={tokenizer.pad_token_id}, UNK={tokenizer.unk_token_id}, BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}")

# Load test data
print("\nLoading test data...")
test_dataset = load_dataset("json", data_dir="./processed_data/splits", split='test')
print(f"Test samples: {len(test_dataset)}")

# Load item mapping
with open("./processed_data/item_mapping.json", 'r') as f:
    item_mapping = json.load(f)
num_items = item_mapping['num_items']
print(f"Number of items: {num_items}")

# Sample some test cases
np.random.seed(42)
sample_indices = np.random.choice(len(test_dataset), size=10, replace=False)

print("\n" + "=" * 60)
print("Analyzing Sample Predictions")
print("=" * 60)

with torch.no_grad():
    for idx in sample_indices:
        sample = test_dataset[int(idx)]
        prompt = sample['prompt']
        groundtruth = sample['ground_truth']

        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # Get logits
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :]  # Last position logits

        # Get top predictions
        # Item IDs start from 4 (after special tokens)
        item_logits = logits[4:]  # Skip special tokens
        top_k_scores, top_k_indices = torch.topk(item_logits, k=20, dim=-1)
        top_k_indices = top_k_indices + 4  # Shift back

        # Convert to strings
        top_k_preds = [str(idx.item()) for idx in top_k_indices]

        # Check if groundtruth in top-k
        in_top1 = groundtruth == top_k_preds[0]
        in_top5 = groundtruth in top_k_preds[:5]
        in_top10 = groundtruth in top_k_preds[:10]
        in_top20 = groundtruth in top_k_preds[:20]

        print(f"\n--- Sample {idx} ---")
        print(f"Prompt (last 5 items): {' '.join(prompt.split()[-5:])}")
        print(f"Groundtruth: {groundtruth}")
        print(f"Top-5 predictions: {top_k_preds[:5]}")
        print(f"Top-1: {in_top1}, Top-5: {in_top5}, Top-10: {in_top10}, Top-20: {in_top20}")

        # Check groundtruth rank
        if groundtruth in top_k_preds:
            rank = top_k_preds.index(groundtruth) + 1
            print(f"Groundtruth rank: {rank}")
        else:
            # Check logit for groundtruth
            gt_id = int(groundtruth) + 4  # Shift for special tokens
            gt_logit = logits[gt_id].item()
            gt_rank = (logits > logits[gt_id]).sum().item() + 1
            print(f"Groundtruth not in top-20. Logit: {gt_logit:.4f}, Rank: {gt_rank}")

# Analyze prediction distribution
print("\n" + "=" * 60)
print("Analyzing Prediction Distribution (Full Test Set)")
print("=" * 60)

all_top1_preds = []
all_top20_preds = []
all_groundtruths = []

with torch.no_grad():
    for i, sample in enumerate(test_dataset):
        if i >= 500:  # Sample 500 for speed
            break

        prompt = sample['prompt']
        groundtruth = sample['ground_truth']
        all_groundtruths.append(groundtruth)

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :]

        item_logits = logits[4:]
        top_k_scores, top_k_indices = torch.topk(item_logits, k=20, dim=-1)
        top_k_indices = top_k_indices + 4

        all_top1_preds.append(str(top_k_indices[0].item()))
        all_top20_preds.extend([str(idx.item()) for idx in top_k_indices])

# Count most common predictions
print("\nMost common Top-1 predictions:")
top1_counter = Counter(all_top1_preds)
for pred, count in top1_counter.most_common(10):
    print(f"  Item {pred}: {count} times ({count/len(all_top1_preds)*100:.1f}%)")

print("\nMost common Top-20 predictions:")
top20_counter = Counter(all_top20_preds)
for pred, count in top20_counter.most_common(10):
    print(f"  Item {pred}: {count} times")

print("\nGroundtruth distribution:")
gt_counter = Counter(all_groundtruths)
print(f"  Unique groundtruth items: {len(gt_counter)}")
print(f"  Most common groundtruths: {gt_counter.most_common(5)}")

# Check overlap
top1_items = set(all_top1_preds)
gt_items = set(all_groundtruths)
overlap = top1_items & gt_items
print(f"\nOverlap between Top-1 predictions and groundtruths: {len(overlap)} items")

# Check if model is just predicting popular items
print("\n" + "=" * 60)
print("Checking if model predicts popular items")
print("=" * 60)

# Load item frequencies from training data
train_data = load_dataset("json", data_dir="./processed_data/splits", split='train')
item_freq = Counter()
for sample in train_data:
    items = sample['prompt'].split()
    item_freq.update(items)
    item_freq[sample['ground_truth']] += 1

print("\nMost popular items in training data:")
for item, count in item_freq.most_common(10):
    print(f"  Item {item}: {count} times")

print("\nModel's most common Top-1 predictions vs popularity:")
for pred, count in top1_counter.most_common(5):
    train_freq = item_freq.get(pred, 0)
    print(f"  Item {pred}: predicted {count} times, train freq: {train_freq}")
