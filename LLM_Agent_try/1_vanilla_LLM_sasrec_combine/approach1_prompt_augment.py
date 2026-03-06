#!/usr/bin/env python3
"""
Approach 1: Prompt Augmentation - Include SASRec predictions in LLM prompt

This approach:
1. Uses SASRec to get top-k candidate items for each user
2. Includes these candidates as additional context in the LLM prompt
3. LLM can use this information to refine its prediction

The hypothesis is that SASRec's sequential pattern recognition can guide
the LLM's semantic understanding towards more relevant predictions.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast

# Add paths for SASRec imports
sys.path.append("../../Rec-Transformer")
sys.path.append("../..")
from sasrec import SasRecForCausalLM, SasRecConfig

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
TITLES_PATH = os.path.join(SCRIPT_DIR, "../use_Qwen3-1-7B_to_generate_title/item_titles_unique.json")
LLM_TEST_DATA = os.path.join(SCRIPT_DIR, "../LLM_Rec_Data_Preparation/test.json")

# SASRec paths
SASREC_CHECKPOINT = os.path.join(SCRIPT_DIR, "../vanilla_sasrec/checkpoints/sasrec_beauty_20260226_055626/best_model")
ITEM_MAPPING_PATH = os.path.join(SCRIPT_DIR, "../vanilla_sasrec/processed_data/item_mapping.json")
SASREC_DATA_PATH = os.path.join(SCRIPT_DIR, "../vanilla_sasrec/processed_data/splits")

# LLM path
LLM_MODEL_PATH = os.path.join(SCRIPT_DIR, "../../LLM4RecPart/models/Qwen3-1-7B")

# Evaluation settings
K_VALUES = [1, 5, 10, 20]
MAX_HISTORY_LEN = 20
MAX_REVIEW_LEN = 150
BATCH_SIZE = 8

# Number of SASRec candidates to include in prompt
SASREC_CANDIDATE_KS = [5, 10, 20]  # Try different numbers of candidates


# =============================================================================
# Data Loading
# =============================================================================
def load_item_titles():
    """Load condensed item titles."""
    print("Loading item titles...")
    with open(TITLES_PATH, 'r', encoding='utf-8') as f:
        titles_data = json.load(f)

    titles = {}
    for item in titles_data:
        titles[item['item_id']] = item['condensed_title']

    print(f"  Loaded {len(titles)} item titles")
    return titles


def load_item_mapping():
    """Load item_id <-> internal_id mapping."""
    print("Loading item mapping...")
    with open(ITEM_MAPPING_PATH, 'r') as f:
        mapping_data = json.load(f)

    item_to_id = mapping_data['item_to_id']
    id_to_item = {v: k for k, v in item_to_id.items()}
    num_items = mapping_data['num_items']

    print(f"  Loaded mapping for {num_items} items")
    return item_to_id, id_to_item, num_items


def load_test_data():
    """Load test data."""
    print("Loading LLM test data...")
    with open(LLM_TEST_DATA, 'r') as f:
        llm_test = json.load(f)

    print("Loading SASRec test data...")
    with open(os.path.join(SASREC_DATA_PATH, "test.json"), 'r') as f:
        sasrec_test = json.load(f)

    print(f"  LLM test: {len(llm_test)} samples")
    print(f"  SASRec test: {len(sasrec_test)} samples")

    return llm_test, sasrec_test


# =============================================================================
# Model Loading
# =============================================================================
def load_sasrec_model():
    """Load SASRec model and tokenizer."""
    print(f"Loading SASRec model from {SASREC_CHECKPOINT}...")

    model = SasRecForCausalLM.from_pretrained(SASREC_CHECKPOINT)
    model = model.cuda()
    model.eval()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(SASREC_CHECKPOINT)
    tokenizer.padding_side = 'left'

    print(f"  SASRec model loaded")
    return model, tokenizer


def load_llm_model():
    """Load LLM model and tokenizer."""
    print(f"Loading LLM model from {LLM_MODEL_PATH}...")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()
    model.eval()

    print(f"  LLM model loaded")
    return model, tokenizer


# =============================================================================
# Prompt Augmentation
# =============================================================================
def create_augmented_prompt(history_items, item_titles, sasrec_candidates, num_candidates=10):
    """
    Create LLM prompt augmented with SASRec candidates.

    Args:
        history_items: List of (item_id, rating, review_text) tuples
        item_titles: Dict of item_id -> title
        sasrec_candidates: List of (item_id, score) tuples from SASRec
        num_candidates: Number of candidates to include
    """
    # Build history section
    history_lines = []
    for i, (item_id, rating, review_text) in enumerate(history_items, 1):
        title = item_titles.get(item_id, "Unknown Product")
        truncated_review = review_text[:MAX_REVIEW_LEN] if len(review_text) > MAX_REVIEW_LEN else review_text
        if len(review_text) > MAX_REVIEW_LEN:
            truncated_review += "..."
        history_lines.append(f'{i}. "{title}" (Rating: {int(rating)}) - Review: {truncated_review}')

    history_text = "\n".join(history_lines)

    # Build SASRec candidates section
    candidate_lines = []
    for i, (item_id, score) in enumerate(sasrec_candidates[:num_candidates], 1):
        title = item_titles.get(item_id, "Unknown Product")
        candidate_lines.append(f'{i}. "{title}" (Score: {score:.3f})')

    candidates_text = "\n".join(candidate_lines)

    # Create augmented prompt
    prompt = f"""User's purchase history:
{history_text}

Based on a sequential pattern analysis, here are some candidate products the user might be interested in:
{candidates_text}

Considering both the user's interaction history and the candidate suggestions above, predict the next product they would be most interested in purchasing.

Next product:"""

    return prompt


def get_sasrec_top_k(model, tokenizer, item_sequence, id_to_item, item_titles, k=20):
    """
    Get top-k items from SASRec with titles.

    Returns:
        List of (original_item_id, score) tuples
    """
    seq_str = " ".join(str(item_id) for item_id in item_sequence)
    inputs = tokenizer(seq_str, return_tensors='pt')
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

    # Get scores for all items (exclude special tokens 0-3)
    item_logits = logits[4:]
    probs = F.softmax(item_logits, dim=0)

    top_k_scores, top_k_indices = torch.topk(probs, k)

    candidates = []
    for internal_id, score in zip(top_k_indices.tolist(), top_k_scores.tolist()):
        if internal_id in id_to_item:
            original_id = id_to_item[internal_id]
            candidates.append((original_id, score))

    return candidates


def truncate_review(review_text, max_len=MAX_REVIEW_LEN):
    if len(review_text) <= max_len:
        return review_text
    return review_text[:max_len-3] + "..."


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_with_augmentation(llm_model, llm_tokenizer, item_titles,
                               aligned_samples, sasrec_candidates_dict,
                               num_candidates=10):
    """Evaluate LLM with SASRec-augmented prompts."""

    hr_scores = {k: [] for k in K_VALUES}
    ndcg_scores = {k: [] for k in K_VALUES}
    predictions_log = []

    for sample in tqdm(aligned_samples, desc=f"Eval (k={num_candidates})"):
        llm_sample = sample['llm_sample']
        target_item_id = sample['target_item_id']
        user_id = sample['user_id']

        # Get SASRec candidates
        sasrec_candidates = sasrec_candidates_dict.get(user_id, [])

        # Parse history from LLM sample
        prompt_text = llm_sample['prompt']

        # Extract history items from prompt (for creating augmented prompt)
        # We need to parse the original history to get item_ids
        # Actually, let's use the original sample structure

        # Create augmented prompt
        # First, we need to extract history items from the original prompt
        # or use a pre-built structure

        # For simplicity, let's rebuild the history from the sample
        # We'll need to modify the data loading to include history item_ids

        # Alternative: Use the existing prompt and just add candidates section
        # This is simpler and avoids parsing issues

        # Build candidates section
        candidate_lines = []
        for i, (item_id, score) in enumerate(sasrec_candidates[:num_candidates], 1):
            title = item_titles.get(item_id, "Unknown Product")
            candidate_lines.append(f'{i}. "{title}" (Score: {score:.3f})')

        candidates_text = "\n".join(candidate_lines)

        # Create augmented prompt by inserting candidates
        augmented_prompt = f"""{prompt_text}

Based on sequential pattern analysis, here are some candidate products:
{candidates_text}

Which of these (or another product) would the user most likely purchase next?

Next product:"""

        # Generate prediction
        inputs = llm_tokenizer(augmented_prompt, return_tensors='pt').to('cuda')

        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=llm_tokenizer.pad_token_id,
                eos_token_id=llm_tokenizer.eos_token_id,
            )

        input_len = inputs['input_ids'].shape[1]
        predicted_text = llm_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        # Match prediction to items using string similarity
        from difflib import SequenceMatcher

        similarities = []
        for item_id, title in item_titles.items():
            sim = SequenceMatcher(None, predicted_text.lower(), title.lower()).ratio()
            similarities.append((item_id, sim, title))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_items = similarities[:max(K_VALUES)]

        # Compute metrics
        for k in K_VALUES:
            top_k_ids = [item[0] for item in top_k_items[:k]]
            if target_item_id in top_k_ids:
                hr_scores[k].append(1.0)
                rank = top_k_ids.index(target_item_id) + 1
                ndcg_scores[k].append(1.0 / np.log2(rank + 1))
            else:
                hr_scores[k].append(0.0)
                ndcg_scores[k].append(0.0)

        # Log first few
        if len(predictions_log) < 5:
            predictions_log.append({
                'target': item_titles.get(target_item_id, 'Unknown')[:50],
                'predicted': predicted_text[:50],
                'top_5': [(item_titles.get(s[0], '?')[:30], f"{s[1]:.3f}") for s in top_k_items[:5]]
            })

    results = {
        'num_candidates': num_candidates,
        'HR': {f'HR@{k}': np.mean(scores) for k, scores in hr_scores.items()},
        'NDCG': {f'NDCG@{k}': np.mean(scores) for k, scores in ndcg_scores.items()},
        'predictions_log': predictions_log
    }

    return results


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("Approach 1: Prompt Augmentation (SASRec candidates -> LLM)")
    print("=" * 60)

    # Step 1: Load data
    item_titles = load_item_titles()
    item_to_id, id_to_item, num_items = load_item_mapping()
    llm_test, sasrec_test = load_test_data()

    # Step 2: Align test samples
    print("\nAligning test samples...")

    llm_test_lookup = {}
    for sample in llm_test:
        key = (sample['user_id'], sample['target_item_id'])
        llm_test_lookup[key] = sample

    sasrec_test_lookup = {}
    for sample in sasrec_test:
        key = (sample['user_id'], sample['ground_truth'])
        sasrec_test_lookup[key] = sample

    aligned_samples = []
    for llm_sample in llm_test:
        user_id = llm_sample['user_id']
        target_item_id = llm_sample['target_item_id']

        if target_item_id not in item_to_id:
            continue

        target_internal_id = str(item_to_id[target_item_id])
        sasrec_key = (user_id, target_internal_id)

        if sasrec_key in sasrec_test_lookup:
            sasrec_sample = sasrec_test_lookup[sasrec_key]
            aligned_samples.append({
                'llm_sample': llm_sample,
                'sasrec_sample': sasrec_sample,
                'user_id': user_id,
                'target_item_id': target_item_id,
                'target_internal_id': target_internal_id
            })

    print(f"  Aligned samples: {len(aligned_samples)}")

    if len(aligned_samples) == 0:
        print("ERROR: No aligned samples!")
        return

    # Step 3: Load models
    print("\nLoading models...")
    sasrec_model, sasrec_tokenizer = load_sasrec_model()
    llm_model, llm_tokenizer = load_llm_model()

    # Step 4: Pre-compute SASRec candidates for all samples
    print("\nPre-computing SASRec candidates...")

    sasrec_candidates_dict = {}
    for sample in tqdm(aligned_samples, desc="SASRec inference"):
        user_id = sample['user_id']
        sasrec_sample = sample['sasrec_sample']

        if user_id in sasrec_candidates_dict:
            continue

        sasrec_prompt = sasrec_sample['prompt']
        sasrec_seq = [int(x) for x in sasrec_prompt.split()]

        candidates = get_sasrec_top_k(
            sasrec_model, sasrec_tokenizer, sasrec_seq,
            id_to_item, item_titles, k=30
        )
        sasrec_candidates_dict[user_id] = candidates

    print(f"  Pre-computed candidates for {len(sasrec_candidates_dict)} users")

    # Step 5: Evaluate with different numbers of candidates
    print("\n" + "=" * 60)
    print("Evaluating Prompt Augmentation...")
    print("=" * 60)

    results_by_k = {}

    for num_cand in SASREC_CANDIDATE_KS:
        print(f"\n--- Testing with {num_cand} SASRec candidates ---")

        results = evaluate_with_augmentation(
            llm_model, llm_tokenizer, item_titles,
            aligned_samples, sasrec_candidates_dict,
            num_candidates=num_cand
        )
        results_by_k[num_cand] = results

        print(f"\nResults (num_candidates={num_cand}):")
        for k in K_VALUES:
            print(f"  HR@{k}: {results['HR'][f'HR@{k}']:.4f}")
            print(f"  NDCG@{k}: {results['NDCG'][f'NDCG@{k}']:.4f}")

        # Print sample predictions
        if results['predictions_log']:
            print("\nSample predictions:")
            for i, pred in enumerate(results['predictions_log'][:3]):
                print(f"  {i+1}. Target: {pred['target']}")
                print(f"     Predicted: {pred['predicted']}")

    # Step 6: Print final comparison
    print("\n" + "=" * 60)
    print("Final Results Comparison")
    print("=" * 60)

    print("\nBaseline:")
    print("  SASRec alone:    HR@1=9.22%, HR@10=13.03%")
    print("  LLM alone:       HR@1=9.26%, HR@10=12.18%")

    print("\nPrompt Augmentation Results:")
    for num_cand, results in sorted(results_by_k.items()):
        print(f"\n  Num candidates={num_cand}:")
        for k in [1, 10]:
            print(f"    HR@{k}: {results['HR'][f'HR@{k}']:.4f} ({results['HR'][f'HR@{k}']*100:.2f}%)")

    # Step 7: Save results
    output_path = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_path, exist_ok=True)

    final_results = {
        'candidate_ks': SASREC_CANDIDATE_KS,
        'results_by_k': {str(k): {
            'num_candidates': v['num_candidates'],
            'HR': v['HR'],
            'NDCG': v['NDCG']
        } for k, v in results_by_k.items()},
        'num_test_samples': len(aligned_samples),
        'baselines': {
            'sasrec': {'HR@1': 0.0922, 'HR@10': 0.1303},
            'llm': {'HR@1': 0.0926, 'HR@10': 0.1218}
        }
    }

    with open(os.path.join(output_path, "prompt_augment_results.json"), 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to {output_path}/prompt_augment_results.json")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
