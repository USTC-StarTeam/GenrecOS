#!/usr/bin/env python3
"""
Approach 2: Score Difference-based Adaptive Fusion

Key idea: Use the difference between LLM similarity and SASRec score to determine fusion.
When the two scores disagree significantly, we need to decide which to trust more.

Strategies to explore:
1. When scores disagree (high diff) -> trust the higher score (more confident model)
2. When scores disagree (high diff) -> trust the lower score (conservative approach)
3. When scores agree (low diff) -> use equal weighting
4. Dynamic alpha based on relative confidence

Hypothesis:
- If SASRec score >> LLM similarity: SASRec found a strong pattern, trust it
- If LLM similarity >> SASRec score: LLM found semantic match, trust it
- If both agree: confident prediction, either should work
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
BATCH_SIZE = 8


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

    vocab_size = model.config.vocab_size

    print(f"  SASRec model loaded: {model.num_parameters() / 1e6:.2f}M parameters")
    return model, tokenizer, vocab_size


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
# Embedding Computation
# =============================================================================
def get_title_embeddings(model, tokenizer, item_ids, item_titles, batch_size=64):
    """Pre-compute embeddings for all item titles."""
    print(f"\nComputing embeddings for {len(item_ids)} items...")

    embeddings = []
    item_list = list(item_ids)

    for i in tqdm(range(0, len(item_list), batch_size), desc="Encoding titles"):
        batch_items = item_list[i:i+batch_size]
        batch_titles = [item_titles.get(item, "Unknown") for item in batch_items]

        inputs = tokenizer(batch_titles, return_tensors='pt', padding=True, truncation=True, max_length=50)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.model(**inputs, use_cache=False)
            hidden_states = outputs.last_hidden_state

            attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
            masked_hidden = hidden_states * attention_mask
            batch_embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)

        embeddings.append(batch_embeddings.cpu())

    all_embeddings = torch.cat(embeddings, dim=0)
    item_to_idx = {item: idx for idx, item in enumerate(item_list)}

    return all_embeddings.cuda(), item_to_idx, item_list


def get_prediction_embedding(model, tokenizer, prediction_text):
    """Get embedding for a single prediction text."""
    inputs = tokenizer([prediction_text], return_tensors='pt', padding=True, truncation=True, max_length=50)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.model(**inputs, use_cache=False)
        hidden_states = outputs.last_hidden_state

        attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
        masked_hidden = hidden_states * attention_mask
        embedding = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)

    return embedding.squeeze(0)


# =============================================================================
# SASRec Scoring
# =============================================================================
def get_sasrec_scores_for_sequence(model, tokenizer, item_sequence, vocab_size, max_seq_length=100):
    """Get SASRec scores for a sequence of items."""
    if len(item_sequence) > max_seq_length:
        item_sequence = item_sequence[-max_seq_length:]

    seq_str = " ".join(str(item_id) for item_id in item_sequence)
    inputs = tokenizer(seq_str, return_tensors='pt')
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

    num_items = vocab_size - 4
    item_logits = logits[4:]
    probs = F.softmax(item_logits, dim=0)

    return probs


# =============================================================================
# Score Difference-based Fusion Strategies
# =============================================================================
def strategy_trust_higher(sasrec_score, llm_score, diff, base_alpha=0.5):
    """
    Strategy 1: Trust the higher score when they disagree.

    If sasrec > llm -> increase alpha (trust SASRec)
    If llm > sasrec -> decrease alpha (trust LLM)
    """
    # Normalize difference to [-1, 1]
    normalized_diff = torch.tanh(diff * 5)  # Scale up for sensitivity

    # Adjust alpha: positive diff means sasrec > llm -> increase alpha
    alpha = base_alpha + 0.3 * normalized_diff

    # Clamp
    alpha = torch.clamp(alpha, 0.1, 0.9)

    return alpha.item()


def strategy_trust_lower(sasrec_score, llm_score, diff, base_alpha=0.5):
    """
    Strategy 2: Trust the lower score (conservative approach).

    If sasrec > llm -> decrease alpha (trust LLM's lower score)
    If llm > sasrec -> increase alpha (trust SASRec's lower score)
    """
    normalized_diff = torch.tanh(diff * 5)
    alpha = base_alpha - 0.3 * normalized_diff
    alpha = torch.clamp(alpha, 0.1, 0.9)

    return alpha.item()


def strategy_dynamic_confidence(sasrec_score, llm_score, diff, base_alpha=0.5):
    """
    Strategy 3: Weight by relative confidence.

    Use the relative magnitude of scores to determine weight.
    Higher score = more confident = more weight.
    """
    # Compute relative confidence
    total = sasrec_score + llm_score + 1e-6
    sasrec_confidence = sasrec_score / total
    llm_confidence = llm_score / total

    # Blend with base_alpha
    alpha = 0.3 * base_alpha + 0.7 * sasrec_confidence
    alpha = torch.clamp(alpha, 0.1, 0.9)

    return alpha.item()


def strategy_adaptive_by_agreement(sasrec_score, llm_score, diff, base_alpha=0.5,
                                   diff_threshold=0.1):
    """
    Strategy 4: Adaptive based on agreement level.

    - Low diff (agreement): Use base_alpha
    - High diff (disagreement): Trust the higher score
    """
    if abs(diff) < diff_threshold:
        # Agreement - use base alpha
        return base_alpha
    else:
        # Disagreement - trust higher score
        normalized_diff = torch.tanh(diff * 5)
        alpha = base_alpha + 0.3 * normalized_diff
        alpha = torch.clamp(alpha, 0.1, 0.9)
        return alpha.item()


def apply_fusion_strategy(sasrec_probs, llm_normalized, strategy_name, base_alpha=0.5):
    """
    Apply a fusion strategy across all items.

    Returns:
        fused_scores: Fused scores tensor
        alpha_values: List of alpha values used per item
    """
    # Compute differences
    diff = sasrec_probs - llm_normalized

    # Select strategy
    if strategy_name == 'trust_higher':
        strategy_fn = strategy_trust_higher
    elif strategy_name == 'trust_lower':
        strategy_fn = strategy_trust_lower
    elif strategy_name == 'dynamic_confidence':
        strategy_fn = strategy_dynamic_confidence
    elif strategy_name == 'adaptive_by_agreement':
        strategy_fn = strategy_adaptive_by_agreement
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Apply strategy per item
    fused_scores = torch.zeros_like(sasrec_probs)
    alpha_values = []

    for i in range(len(sasrec_probs)):
        alpha = strategy_fn(sasrec_probs[i], llm_normalized[i], diff[i], base_alpha)
        alpha_values.append(alpha)
        fused_scores[i] = alpha * sasrec_probs[i] + (1 - alpha) * llm_normalized[i]

    return fused_scores, alpha_values


def find_top_k_fused(fused_scores, item_list, k=20):
    """Find top-k items from fused scores."""
    top_k_values, top_k_indices = torch.topk(fused_scores, k)

    results = []
    for idx, score in zip(top_k_indices.tolist(), top_k_values.tolist()):
        results.append((item_list[idx], score))

    return results


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("Approach 2: Score Difference-based Adaptive Fusion")
    print("=" * 60)

    # Step 1: Load data
    item_titles = load_item_titles()
    item_to_id, id_to_item, num_items = load_item_mapping()
    llm_test, sasrec_test = load_test_data()

    # Step 2: Align test samples
    print("\nAligning test samples...")

    llm_test_lookup = {(s['user_id'], s['target_item_id']): s for s in llm_test}
    sasrec_test_lookup = {(s['user_id'], s['ground_truth']): s for s in sasrec_test}

    aligned_samples = []
    for llm_sample in llm_test:
        user_id = llm_sample['user_id']
        target_item_id = llm_sample['target_item_id']

        if target_item_id not in item_to_id:
            continue

        target_internal_id = str(item_to_id[target_item_id])
        sasrec_key = (user_id, target_internal_id)

        if sasrec_key in sasrec_test_lookup:
            aligned_samples.append({
                'llm_sample': llm_sample,
                'sasrec_sample': sasrec_test_lookup[sasrec_key],
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
    sasrec_model, sasrec_tokenizer, sasrec_vocab_size = load_sasrec_model()
    llm_model, llm_tokenizer = load_llm_model()

    # Step 4: Pre-compute item embeddings
    valid_items = set(item_to_id.keys())
    item_embeddings, item_to_emb_idx, item_emb_list = get_title_embeddings(
        llm_model, llm_tokenizer, valid_items, item_titles, batch_size=64
    )

    print(f"  Item embeddings shape: {item_embeddings.shape}")

    # Step 5: First pass - generate predictions and collect score statistics
    print("\n" + "=" * 60)
    print("Phase 1: Computing scores for all samples...")
    print("=" * 60)

    all_sasrec_scores = []
    all_llm_scores = []
    all_differences = []
    generation_results = []

    for aligned in tqdm(aligned_samples, desc="Computing scores"):
        llm_sample = aligned['llm_sample']
        sasrec_sample = aligned['sasrec_sample']

        # Get SASRec scores
        sasrec_prompt = sasrec_sample['prompt']
        sasrec_seq = [int(x) for x in sasrec_prompt.split()]
        sasrec_probs = get_sasrec_scores_for_sequence(
            sasrec_model, sasrec_tokenizer, sasrec_seq, sasrec_vocab_size
        )

        # Generate LLM prediction
        llm_prompt = llm_sample['prompt']
        inputs = llm_tokenizer(llm_prompt, return_tensors='pt').to('cuda')

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

        # Get LLM prediction embedding
        pred_embedding = get_prediction_embedding(llm_model, llm_tokenizer, predicted_text)

        # Compute LLM similarity
        pred_norm = F.normalize(pred_embedding.unsqueeze(0), dim=1)
        item_norms = F.normalize(item_embeddings, dim=1)
        llm_similarities = torch.mm(pred_norm, item_norms.T).squeeze(0)

        # Normalize LLM similarity to [0, 1]
        llm_min = llm_similarities.min()
        llm_max = llm_similarities.max()
        if llm_max - llm_min > 1e-6:
            llm_normalized = (llm_similarities - llm_min) / (llm_max - llm_min)
        else:
            llm_normalized = torch.ones_like(llm_similarities) * 0.5

        # Collect statistics (for top items)
        emb_idx_to_internal = {}
        for emb_idx, orig_id in enumerate(item_emb_list):
            if orig_id in item_to_id:
                emb_idx_to_internal[emb_idx] = item_to_id[orig_id]

        # Get top-10 scores for statistics
        for emb_idx in range(min(10, len(item_emb_list))):
            internal_id = emb_idx_to_internal.get(emb_idx)
            if internal_id is not None and internal_id < len(sasrec_probs):
                all_sasrec_scores.append(sasrec_probs[internal_id].item())
                all_llm_scores.append(llm_normalized[emb_idx].item())
                all_differences.append(sasrec_probs[internal_id].item() - llm_normalized[emb_idx].item())

        generation_results.append({
            'predicted_text': predicted_text,
            'sasrec_probs': sasrec_probs,
            'llm_similarities': llm_similarities,
            'llm_normalized': llm_normalized
        })

    # Print statistics
    print(f"\nScore Statistics (top-10 items per sample):")
    print(f"  SASRec scores - Mean: {np.mean(all_sasrec_scores):.4f}, Std: {np.std(all_sasrec_scores):.4f}")
    print(f"  LLM scores - Mean: {np.mean(all_llm_scores):.4f}, Std: {np.std(all_llm_scores):.4f}")
    print(f"  Differences - Mean: {np.mean(all_differences):.4f}, Std: {np.std(all_differences):.4f}")
    print(f"  Abs Differences - Mean: {np.mean(np.abs(all_differences)):.4f}")

    # Step 6: Evaluate different strategies
    print("\n" + "=" * 60)
    print("Phase 2: Evaluating Different Fusion Strategies...")
    print("=" * 60)

    strategies = ['trust_higher', 'trust_lower', 'dynamic_confidence', 'adaptive_by_agreement']
    base_alphas = [0.5, 0.7]

    all_results = {}

    for strategy in strategies:
        for base_alpha in base_alphas:
            config_name = f"{strategy}_alpha_{base_alpha}"
            print(f"\n--- Config: {config_name} ---")

            hr_scores = {k: [] for k in K_VALUES}
            ndcg_scores = {k: [] for k in K_VALUES}
            alpha_values_used = []

            for i, aligned in enumerate(tqdm(aligned_samples, desc=f"Eval {config_name}")):
                target_item_id = aligned['target_item_id']
                gen_result = generation_results[i]

                sasrec_probs = gen_result['sasrec_probs']
                llm_normalized = gen_result['llm_normalized']

                # Build aligned score arrays
                emb_idx_to_internal = {}
                for emb_idx, orig_id in enumerate(item_emb_list):
                    if orig_id in item_to_id:
                        emb_idx_to_internal[emb_idx] = item_to_id[orig_id]

                sasrec_aligned = []
                llm_aligned = []
                valid_indices = []

                for emb_idx in range(len(item_emb_list)):
                    internal_id = emb_idx_to_internal.get(emb_idx)
                    if internal_id is not None and internal_id < len(sasrec_probs):
                        sasrec_aligned.append(sasrec_probs[internal_id])
                        llm_aligned.append(llm_normalized[emb_idx])
                        valid_indices.append(emb_idx)

                if sasrec_aligned:
                    sasrec_t = torch.stack(sasrec_aligned)
                    llm_t = torch.stack(llm_aligned)

                    # Apply fusion strategy
                    fused, alphas = apply_fusion_strategy(
                        sasrec_t, llm_t, strategy, base_alpha
                    )

                    alpha_values_used.extend(alphas)

                    # Get top-k
                    valid_items = [item_emb_list[i] for i in valid_indices]
                    top_k_items = find_top_k_fused(fused, valid_items, k=max(K_VALUES))

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
                else:
                    for k in K_VALUES:
                        hr_scores[k].append(0.0)
                        ndcg_scores[k].append(0.0)

            # Compute average metrics
            results = {
                'strategy': strategy,
                'base_alpha': base_alpha,
                'HR': {f'HR@{k}': np.mean(scores) for k, scores in hr_scores.items()},
                'NDCG': {f'NDCG@{k}': np.mean(scores) for k, scores in ndcg_scores.items()},
                'avg_alpha_used': np.mean(alpha_values_used) if alpha_values_used else base_alpha,
                'alpha_std': np.std(alpha_values_used) if alpha_values_used else 0
            }
            all_results[config_name] = results

            print(f"\nResults ({config_name}):")
            for k in K_VALUES:
                print(f"  HR@{k}: {results['HR'][f'HR@{k}']:.4f} ({results['HR'][f'HR@{k}']*100:.2f}%)")
            print(f"  Avg alpha used: {results['avg_alpha_used']:.3f} (std: {results['alpha_std']:.3f})")

    # Step 7: Compare with baselines
    print("\n" + "=" * 60)
    print("Final Results Comparison")
    print("=" * 60)

    print("\nBaselines:")
    print("  SASRec alone:         HR@1=9.22%, HR@10=13.03%")
    print("  LLM alone:            HR@1=9.26%, HR@10=12.18%")
    print("  Score Fusion (α=0.7): HR@1=10.95%, HR@10=13.25%")

    print("\nScore Difference Adaptive Results:")
    for config_name, results in sorted(all_results.items()):
        print(f"\n  {config_name}:")
        print(f"    HR@1: {results['HR']['HR@1']*100:.2f}%")
        print(f"    HR@10: {results['HR']['HR@10']*100:.2f}%")
        print(f"    Avg alpha: {results['avg_alpha_used']:.3f}")

    # Step 8: Save results
    output_path = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_path, exist_ok=True)

    final_results = {
        'approach': 'score_difference_adaptive',
        'score_statistics': {
            'sasrec_mean': float(np.mean(all_sasrec_scores)),
            'sasrec_std': float(np.std(all_sasrec_scores)),
            'llm_mean': float(np.mean(all_llm_scores)),
            'llm_std': float(np.std(all_llm_scores)),
            'diff_mean': float(np.mean(all_differences)),
            'diff_std': float(np.std(all_differences)),
            'abs_diff_mean': float(np.mean(np.abs(all_differences)))
        },
        'results_by_config': all_results,
        'num_test_samples': len(aligned_samples),
        'baselines': {
            'sasrec': {'HR@1': 0.0922, 'HR@10': 0.1303},
            'llm': {'HR@1': 0.0926, 'HR@10': 0.1218},
            'score_fusion_alpha_0.7': {'HR@1': 0.1095, 'HR@10': 0.1325}
        }
    }

    with open(os.path.join(output_path, "score_difference_adaptive_results.json"), 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to {output_path}/score_difference_adaptive_results.json")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
