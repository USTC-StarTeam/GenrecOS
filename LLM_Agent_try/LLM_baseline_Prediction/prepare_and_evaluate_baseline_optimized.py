#!/usr/bin/env python3
"""
Optimized Baseline LLM Recommendation using Original Titles.

Key optimizations:
1. Batch inference - process multiple samples at once
2. Pre-compute item title embeddings
3. Vector similarity (cosine) instead of string matching
4. Use torch for efficient batch computation
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm
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
BATCH_SIZE = 16  # Process multiple samples at once


def load_metadata():
    """Load metadata to get original titles."""
    print("Loading metadata...")
    metadata = {}
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading metadata"):
            try:
                item_data = json.loads(line.strip())
                item_id = item_data.get('parent_asin') or item_data.get('asin')
                if item_id:
                    title = item_data.get('title', 'Unknown Product')
                    # Clean and truncate title
                    title = title[:100] if len(title) > 100 else title
                    metadata[item_id] = {'title': title}
            except json.JSONDecodeError:
                continue
    print(f"  Loaded {len(metadata)} items")
    return metadata


def load_reviews_and_group(metadata):
    """Load reviews and group by user."""
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

    return user_sequences, item_freq


def filter_data(user_sequences, item_freq):
    """Filter low-frequency items."""
    print(f"\nFiltering items with freq < {MIN_ITEM_FREQ}...")
    valid_items = {item for item, freq in item_freq.items() if freq >= MIN_ITEM_FREQ}

    filtered_sequences = {}
    for user_id, seq in user_sequences.items():
        filtered_seq = [(item, ts, rating, review)
                        for item, ts, rating, review in seq
                        if item in valid_items]
        if len(filtered_seq) >= MIN_SEQ_LENGTH:
            filtered_sequences[user_id] = filtered_seq

    print(f"  Users: {len(filtered_sequences)}, Items: {len(valid_items)}")
    return filtered_sequences, valid_items


def truncate_text(text, max_len):
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."


def create_prompt(history_items, metadata):
    """Create prompt using ORIGINAL titles."""
    history_lines = []
    for i, (item_id, rating, review_text) in enumerate(history_items, 1):
        title = metadata.get(item_id, {}).get('title', 'Unknown')
        truncated_title = truncate_text(title, 60)
        truncated_review = truncate_text(review_text, MAX_REVIEW_LEN)
        history_lines.append(f'{i}. "{truncated_title}" (Rating: {int(rating)}) - Review: {truncated_review}')

    history_text = "\n".join(history_lines)
    return f"""User's purchase history:
{history_text}

Based on the user's interaction history, predict the next product they would be most interested in purchasing.

Next product:"""


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
    ).cuda()
    model.eval()

    print("  Model loaded successfully")
    return model, tokenizer


def get_title_embeddings(model, tokenizer, item_ids, item_titles, batch_size=64):
    """
    Pre-compute embeddings for all item titles using the model's hidden states.
    Uses the last token's hidden state as the title embedding.
    """
    print(f"\nComputing embeddings for {len(item_ids)} items...")

    embeddings = []
    item_list = list(item_ids)

    for i in tqdm(range(0, len(item_list), batch_size), desc="Encoding titles"):
        batch_items = item_list[i:i+batch_size]
        batch_titles = [item_titles.get(item, "Unknown") for item in batch_items]

        # Tokenize
        inputs = tokenizer(batch_titles, return_tensors='pt', padding=True, truncation=True, max_length=50)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        # Get hidden states
        with torch.no_grad():
            outputs = model.model(**inputs, use_cache=False)
            # Use mean of last hidden states as embedding
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

            # Mean pooling over non-padding tokens
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            masked_hidden = hidden_states * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)
            count = attention_mask.sum(dim=1).clamp(min=1)
            batch_embeddings = (sum_hidden / count)

        embeddings.append(batch_embeddings.cpu())

    # Stack all embeddings
    all_embeddings = torch.cat(embeddings, dim=0)  # [num_items, hidden_dim]

    # Create mapping from item_id to index
    item_to_idx = {item: idx for idx, item in enumerate(item_list)}

    return all_embeddings, item_to_idx, item_list


def find_top_k_by_embedding(predicted_embedding, item_embeddings, item_list, k=20):
    """
    Find top-k items using cosine similarity.
    predicted_embedding: [hidden_dim]
    item_embeddings: [num_items, hidden_dim]
    """
    # Compute cosine similarity
    predicted_norm = F.normalize(predicted_embedding.unsqueeze(0), dim=1)
    item_norms = F.normalize(item_embeddings, dim=1)
    similarities = torch.mm(predicted_norm, item_norms.T).squeeze(0)  # [num_items]

    # Get top-k
    top_k_values, top_k_indices = torch.topk(similarities, k)

    results = []
    for idx, sim in zip(top_k_indices.tolist(), top_k_values.tolist()):
        results.append((item_list[idx], sim))

    return results


def batch_generate_predictions(model, tokenizer, prompts, max_new_tokens=MAX_NEW_TOKENS):
    """Generate predictions for a batch of prompts."""
    # Tokenize all prompts
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    predictions = []
    for i, output in enumerate(outputs):
        input_len = inputs['input_ids'][i].shape[0]
        generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        predictions.append(generated.strip())

    return predictions


def get_prediction_embeddings(model, tokenizer, predictions):
    """Get embeddings for predicted texts."""
    if not predictions:
        return None

    # Tokenize
    inputs = tokenizer(predictions, return_tensors='pt', padding=True, truncation=True, max_length=50)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.model(**inputs, use_cache=False)
        hidden_states = outputs.last_hidden_state

        # Mean pooling
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        masked_hidden = hidden_states * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        count = attention_mask.sum(dim=1).clamp(min=1)
        embeddings = sum_hidden / count

    return embeddings


def main():
    print("=" * 60)
    print("Optimized Baseline LLM Recommendation")
    print("=" * 60)

    # Step 1: Load metadata
    metadata = load_metadata()
    item_titles = {item_id: data['title'] for item_id, data in metadata.items()}

    # Step 2: Load reviews
    user_sequences, item_freq = load_reviews_and_group(metadata)

    # Step 3: Filter
    filtered_sequences, valid_items = filter_data(user_sequences, item_freq)

    # Step 4: Create test samples
    print("\nCreating test samples...")
    test_samples = []
    for user_id, seq in filtered_sequences.items():
        sorted_seq = sorted(seq, key=lambda x: x[1])
        items = [(item, rating, review) for item, _, rating, review in sorted_seq]

        if len(items) < MIN_SEQ_LENGTH:
            continue

        test_history = items[:-1][-MAX_HISTORY_LEN:]
        target_item = items[-1][0]
        target_title = metadata.get(target_item, {}).get('title', 'Unknown')

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

    # Step 6: Pre-compute item embeddings
    item_embeddings, item_to_idx, item_list = get_title_embeddings(
        model, tokenizer, valid_items, item_titles, batch_size=64
    )
    # Move to GPU for faster similarity computation
    item_embeddings = item_embeddings.cuda()

    print(f"  Item embeddings shape: {item_embeddings.shape}")

    # Step 7: Evaluate with batch processing
    print(f"\nEvaluating with batch_size={BATCH_SIZE}...")

    hr_scores = {k: [] for k in K_VALUES}
    ndcg_scores = {k: [] for k in K_VALUES}
    predictions_log = []

    for batch_start in tqdm(range(0, len(test_samples), BATCH_SIZE), desc="Evaluating"):
        batch_end = min(batch_start + BATCH_SIZE, len(test_samples))
        batch_samples = test_samples[batch_start:batch_end]
        batch_prompts = [s['prompt'] for s in batch_samples]

        # Generate predictions
        batch_predictions = batch_generate_predictions(model, tokenizer, batch_prompts)

        # Get embeddings for predictions
        pred_embeddings = get_prediction_embeddings(model, tokenizer, batch_predictions)

        # Compute metrics for each sample
        for i, sample in enumerate(batch_samples):
            target_item_id = sample['target_item_id']
            predicted_text = batch_predictions[i]
            pred_emb = pred_embeddings[i]

            # Find top-k items by embedding similarity
            ranked_items = find_top_k_by_embedding(pred_emb, item_embeddings, item_list, k=max(K_VALUES))

            # Compute metrics
            for k in K_VALUES:
                top_k_ids = [item[0] for item in ranked_items[:k]]
                if target_item_id in top_k_ids:
                    hr_scores[k].append(1.0)
                    rank = top_k_ids.index(target_item_id) + 1
                    ndcg_scores[k].append(1.0 / np.log2(rank + 1))
                else:
                    hr_scores[k].append(0.0)
                    ndcg_scores[k].append(0.0)

            # Log first few predictions
            if len(predictions_log) < 20:
                predictions_log.append({
                    'target': truncate_text(sample['target_title'], 50),
                    'predicted': predicted_text[:50],
                    'top_5': [(item, f"{sim:.3f}") for item, sim in ranked_items[:5]]
                })

    # Step 8: Compute final results
    results = {
        'HR': {f'HR@{k}': np.mean(scores) for k, scores in hr_scores.items()},
        'NDCG': {f'NDCG@{k}': np.mean(scores) for k, scores in ndcg_scores.items()}
    }

    # Step 9: Print results
    print("\n" + "=" * 60)
    print("Evaluation Results (Optimized Baseline - Original Titles)")
    print("=" * 60)

    print("\nHit Rate:")
    for metric, value in results['HR'].items():
        print(f"  {metric}: {value:.4f}")

    print("\nNDCG:")
    for metric, value in results['NDCG'].items():
        print(f"  {metric}: {value:.4f}")

    # Step 10: Save results
    output_results = {
        'metrics': {**results['HR'], **results['NDCG']},
        'num_test_samples': len(test_samples),
        'num_items': len(valid_items),
        'method': 'original_titles_optimized',
        'batch_size': BATCH_SIZE
    }

    with open(os.path.join(SCRIPT_DIR, "baseline_results_optimized.json"), 'w') as f:
        json.dump(output_results, f, indent=2)
    print(f"\nResults saved to baseline_results_optimized.json")

    # Print sample predictions
    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)
    for i, pred in enumerate(predictions_log[:5]):
        print(f"\n{i+1}. Target: {pred['target']}")
        print(f"   Predicted: {pred['predicted']}")
        print(f"   Top 5: {pred['top_5']}")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
