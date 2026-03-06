#!/usr/bin/env python3
"""
Evaluate Qwen model on recommendation task without fine-tuning.

This script:
1. Loads the test set
2. Uses Qwen model to predict next item title
3. Matches predicted title against all item titles
4. Calculates HR@K and NDCG@K metrics
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from difflib import SequenceMatcher

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "../../LLM4RecPart/models/Qwen3-1-7B")
TITLES_PATH = os.path.join(SCRIPT_DIR, "../use_Qwen3-1-7B_to_generate_title/item_titles_unique.json")
TEST_PATH = os.path.join(SCRIPT_DIR, "test.json")

# Evaluation settings
K_VALUES = [1, 5, 10, 20]
MAX_NEW_TOKENS = 20  # Short generation for title prediction
BATCH_SIZE = 1  # Process one at a time for accuracy


def load_data():
    """Load test data and item titles."""
    print("Loading data...")

    # Load test samples
    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"  Test samples: {len(test_data)}")

    # Load all item titles for candidate matching
    with open(TITLES_PATH, 'r', encoding='utf-8') as f:
        titles_data = json.load(f)

    # Create lookup: item_id -> title
    item_to_title = {item['item_id']: item['condensed_title'] for item in titles_data}
    title_to_items = {}
    for item in titles_data:
        title = item['condensed_title']
        if title not in title_to_items:
            title_to_items[title] = []
        title_to_items[title].append(item['item_id'])

    print(f"  Unique items: {len(item_to_title)}")
    print(f"  Unique titles: {len(title_to_items)}")

    return test_data, item_to_title, title_to_items


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
            temperature=0.1,  # Low temperature for more deterministic output
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def string_similarity(s1, s2):
    """Calculate string similarity using SequenceMatcher."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def find_top_k_items(predicted_title, item_to_title, k=20):
    """
    Find top-k items that best match the predicted title.
    Returns list of (item_id, similarity_score) sorted by similarity.
    """
    similarities = []
    for item_id, title in item_to_title.items():
        sim = string_similarity(predicted_title, title)
        similarities.append((item_id, sim, title))

    # Sort by similarity descending
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


def evaluate(model, tokenizer, test_data, item_to_title):
    """Evaluate model on test set."""
    print(f"\nEvaluating on {len(test_data)} test samples...")

    hr_scores = {k: [] for k in K_VALUES}
    ndcg_scores = {k: [] for k in K_VALUES}
    predictions = []

    for sample in tqdm(test_data, desc="Evaluating"):
        prompt = sample['prompt']
        target_item_id = sample['target_item_id']
        target_title = sample['target_title']

        # Generate prediction
        predicted_text = generate_prediction(model, tokenizer, prompt)

        # Find top-k matching items
        ranked_items = find_top_k_items(predicted_text, item_to_title, k=max(K_VALUES))

        # Compute metrics
        for k in K_VALUES:
            hr = compute_hr_at_k(ranked_items, target_item_id, k)
            ndcg = compute_ndcg_at_k(ranked_items, target_item_id, k)
            hr_scores[k].append(hr)
            ndcg_scores[k].append(ndcg)

        # Store prediction for analysis
        predictions.append({
            'user_id': sample['user_id'],
            'target_item_id': target_item_id,
            'target_title': target_title,
            'predicted_text': predicted_text,
            'top_5_items': [(item[0], item[2], f"{item[1]:.3f}") for item in ranked_items[:5]]
        })

    # Compute average metrics
    results = {
        'HR': {f'HR@{k}': np.mean(scores) for k, scores in hr_scores.items()},
        'NDCG': {f'NDCG@{k}': np.mean(scores) for k, scores in ndcg_scores.items()}
    }

    return results, predictions


def main():
    print("=" * 60)
    print("Qwen Model Evaluation on Recommendation Task")
    print("=" * 60)

    # Load data
    test_data, item_to_title, title_to_items = load_data()

    # Load model
    model, tokenizer = load_model()

    # Evaluate
    results, predictions = evaluate(model, tokenizer, test_data, item_to_title)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print("\nHit Rate:")
    for metric, value in results['HR'].items():
        print(f"  {metric}: {value:.4f}")

    print("\nNDCG:")
    for metric, value in results['NDCG'].items():
        print(f"  {metric}: {value:.4f}")

    # Save results
    output_results = {
        'metrics': {**results['HR'], **results['NDCG']},
        'num_test_samples': len(test_data),
        'num_items': len(item_to_title)
    }

    with open(os.path.join(SCRIPT_DIR, "evaluation_results.json"), 'w') as f:
        json.dump(output_results, f, indent=2)
    print(f"\nResults saved to evaluation_results.json")

    # Save sample predictions for analysis
    with open(os.path.join(SCRIPT_DIR, "sample_predictions.json"), 'w', encoding='utf-8') as f:
        json.dump(predictions[:50], f, indent=2, ensure_ascii=False)
    print(f"Sample predictions saved to sample_predictions.json")

    # Print some sample predictions
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
    print("Evaluation Complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
