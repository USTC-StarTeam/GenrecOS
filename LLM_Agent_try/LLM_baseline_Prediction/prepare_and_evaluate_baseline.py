#!/usr/bin/env python3
"""
Baseline LLM Recommendation using Original Titles from Metadata.

This script:
1. Prepares data using ORIGINAL titles from metadata (not condensed titles)
2. Evaluates Qwen model on recommendation task
3. Compares with the condensed title approach

Key difference from LLM_Rec_Data_Preparation:
- Uses original full titles from meta_All_Beauty.jsonl instead of condensed titles
"""

import os
import json
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEWS_PATH = os.path.join(SCRIPT_DIR, "../../Data/Amazons/data/All_Beauty.jsonl")
METADATA_PATH = os.path.join(SCRIPT_DIR, "../../Data/Amazons/data/meta_All_Beauty.jsonl")
MODEL_PATH = os.path.join(SCRIPT_DIR, "../../LLM4RecPart/models/Qwen3-1-7B")

# Evaluation settings
K_VALUES = [1, 5, 10, 20]
MAX_NEW_TOKENS = 20
MAX_HISTORY_LEN = 20
MAX_REVIEW_LEN = 150
MIN_SEQ_LENGTH = 3
MIN_ITEM_FREQ = 5


def load_metadata():
    """Load metadata to get original titles."""
    print("Loading metadata for original titles...")
    metadata = {}
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading metadata"):
            try:
                item_data = json.loads(line.strip())
                item_id = item_data.get('parent_asin') or item_data.get('asin')
                if item_id:
                    metadata[item_id] = {
                        'title': item_data.get('title', 'Unknown Product'),
                        'store': item_data.get('store', 'Unknown')
                    }
            except json.JSONDecodeError:
                continue
    print(f"  Loaded {len(metadata)} items with metadata")
    return metadata


def load_reviews_and_group(metadata):
    """Load reviews and group by user, using original titles."""
    print("\nLoading reviews...")

    user_sequences = defaultdict(list)
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

                if user_id and asin and asin in metadata:
                    user_sequences[user_id].append((asin, timestamp, rating, review_text))
                    item_freq[asin] += 1
            except (json.JSONDecodeError, KeyError):
                continue

    print(f"  Total reviews: {sum(len(seq) for seq in user_sequences.values())}")
    print(f"  Total users: {len(user_sequences)}")
    print(f"  Total items: {len(item_freq)}")

    return user_sequences, item_freq


def filter_data(user_sequences, item_freq):
    """Filter low-frequency items."""
    print(f"\n[Filtering] Removing items with freq < {MIN_ITEM_FREQ}...")

    valid_items = {item for item, freq in item_freq.items() if freq >= MIN_ITEM_FREQ}
    print(f"  Items after filtering: {len(valid_items)}")

    filtered_sequences = {}
    for user_id, seq in user_sequences.items():
        filtered_seq = [(item, ts, rating, review)
                        for item, ts, rating, review in seq
                        if item in valid_items]
        if len(filtered_seq) >= MIN_SEQ_LENGTH:
            filtered_sequences[user_id] = filtered_seq

    print(f"  Users after filtering: {len(filtered_sequences)}")
    return filtered_sequences


def truncate_review(review_text, max_len=MAX_REVIEW_LEN):
    """Truncate review text."""
    if len(review_text) <= max_len:
        return review_text
    return review_text[:max_len-3] + "..."


def truncate_title(title, max_len=80):
    """Truncate long titles."""
    if len(title) <= max_len:
        return title
    return title[:max_len-3] + "..."


def create_prompt(history_items, metadata):
    """Create LLM-friendly prompt using ORIGINAL titles."""
    history_lines = []
    for i, (item_id, rating, review_text) in enumerate(history_items, 1):
        original_title = metadata.get(item_id, {}).get('title', 'Unknown Product')
        truncated_title = truncate_title(original_title)
        truncated_review = truncate_review(review_text)
        history_lines.append(f'{i}. "{truncated_title}" (Rating: {int(rating)}) - Review: {truncated_review}')

    history_text = "\n".join(history_lines)

    prompt = f"""User's purchase history:
{history_text}

Based on the user's interaction history, predict the next product they would be most interested in purchasing.

Next product:"""

    return prompt


def load_model():
    """Load Qwen model and tokenizer."""
    print(f"\nLoading model from {MODEL_PATH}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()

    print("  Model loaded successfully")
    return model, tokenizer


def generate_prediction(model, tokenizer, prompt):
    """Generate next item title prediction."""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def string_similarity(s1, s2):
    """Calculate string similarity using SequenceMatcher."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def find_top_k_items(predicted_title, item_titles, k=20):
    """Find top-k items that best match the predicted title."""
    similarities = []
    for item_id, title in item_titles.items():
        sim = string_similarity(predicted_title, title)
        similarities.append((item_id, sim, title))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


def compute_hr_at_k(ranked_items, target_item_id, k):
    """Compute Hit Rate at K."""
    top_k_items = [item[0] for item in ranked_items[:k]]
    return 1.0 if target_item_id in top_k_items else 0.0


def compute_ndcg_at_k(ranked_items, target_item_id, k):
    """Compute NDCG at K."""
    top_k_items = [item[0] for item in ranked_items[:k]]
    if target_item_id in top_k_items:
        rank = top_k_items.index(target_item_id) + 1
        return 1.0 / np.log2(rank + 1)
    return 0.0


def main():
    print("=" * 60)
    print("Baseline LLM Recommendation with Original Titles")
    print("=" * 60)

    # Step 1: Load metadata (for original titles)
    metadata = load_metadata()

    # Build item_titles lookup for matching
    item_titles = {item_id: data['title'] for item_id, data in metadata.items()}

    # Step 2: Load reviews and group by user
    user_sequences, item_freq = load_reviews_and_group(metadata)

    # Step 3: Filter data
    filtered_sequences = filter_data(user_sequences, item_freq)

    # Step 4: Create test samples (same logic as before - last item for test)
    print("\n[Creating Test Samples]")
    test_samples = []

    for user_id, seq in filtered_sequences.items():
        sorted_seq = sorted(seq, key=lambda x: x[1])
        items = [(item, rating, review) for item, _, rating, review in sorted_seq]

        if len(items) < MIN_SEQ_LENGTH:
            continue

        # Test: last item
        test_history = items[:-1][-MAX_HISTORY_LEN:]
        target_item = items[-1][0]
        target_title = metadata.get(target_item, {}).get('title', 'Unknown Product')

        prompt = create_prompt(test_history, metadata)
        test_samples.append({
            'prompt': prompt,
            'target_item_id': target_item,
            'target_title': target_title,
            'user_id': user_id
        })

    print(f"  Test samples: {len(test_samples)}")

    # Step 5: Load model
    model, tokenizer = load_model()

    # Step 6: Evaluate
    print(f"\n[Evaluating] Processing {len(test_samples)} test samples...")

    hr_scores = {k: [] for k in K_VALUES}
    ndcg_scores = {k: [] for k in K_VALUES}
    predictions = []

    for sample in tqdm(test_samples, desc="Evaluating"):
        prompt = sample['prompt']
        target_item_id = sample['target_item_id']
        target_title = sample['target_title']

        # Generate prediction
        predicted_text = generate_prediction(model, tokenizer, prompt)

        # Find top-k matching items
        ranked_items = find_top_k_items(predicted_text, item_titles, k=max(K_VALUES))

        # Compute metrics
        for k in K_VALUES:
            hr = compute_hr_at_k(ranked_items, target_item_id, k)
            ndcg = compute_ndcg_at_k(ranked_items, target_item_id, k)
            hr_scores[k].append(hr)
            ndcg_scores[k].append(ndcg)

        # Store prediction
        predictions.append({
            'user_id': sample['user_id'],
            'target_item_id': target_item_id,
            'target_title': truncate_title(target_title),
            'predicted_text': predicted_text,
            'top_5_items': [(item[0], truncate_title(item[2]), f"{item[1]:.3f}") for item in ranked_items[:5]]
        })

    # Compute average metrics
    results = {
        'HR': {f'HR@{k}': np.mean(scores) for k, scores in hr_scores.items()},
        'NDCG': {f'NDCG@{k}': np.mean(scores) for k, scores in ndcg_scores.items()}
    }

    # Step 7: Print results
    print("\n" + "=" * 60)
    print("Evaluation Results (Baseline - Original Titles)")
    print("=" * 60)

    print("\nHit Rate:")
    for metric, value in results['HR'].items():
        print(f"  {metric}: {value:.4f}")

    print("\nNDCG:")
    for metric, value in results['NDCG'].items():
        print(f"  {metric}: {value:.4f}")

    # Step 8: Save results
    output_results = {
        'metrics': {**results['HR'], **results['NDCG']},
        'num_test_samples': len(test_samples),
        'num_items': len(item_titles),
        'method': 'original_titles'
    }

    with open(os.path.join(SCRIPT_DIR, "baseline_results.json"), 'w') as f:
        json.dump(output_results, f, indent=2)
    print(f"\nResults saved to baseline_results.json")

    # Save sample predictions
    with open(os.path.join(SCRIPT_DIR, "baseline_predictions.json"), 'w', encoding='utf-8') as f:
        json.dump(predictions[:50], f, indent=2, ensure_ascii=False)
    print(f"Sample predictions saved to baseline_predictions.json")

    # Print sample predictions
    print("\n" + "=" * 60)
    print("Sample Predictions (first 5)")
    print("=" * 60)
    for i, pred in enumerate(predictions[:5]):
        print(f"\nSample {i+1}:")
        print(f"  Target: \"{pred['target_title']}\" ({pred['target_item_id']})")
        print(f"  Predicted: \"{pred['predicted_text']}\"")
        print(f"  Top 5 retrieved:")
        for j, (item_id, title, sim) in enumerate(pred['top_5_items'], 1):
            match = "✓" if item_id == pred['target_item_id'] else " "
            print(f"    {j}. [{match}] {title} (sim={sim})")

    print("\n" + "=" * 60)
    print("Baseline Evaluation Complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
