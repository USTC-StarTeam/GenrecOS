#!/usr/bin/env python3
"""
Approach 2: Score Fusion - Combine SASRec scores with LLM embedding similarity

This approach:
1. Uses SASRec to get preliminary ranking scores for all candidate items
2. Uses LLM to generate prediction embedding and compute similarity with all items
3. Fuses both scores: final_score = alpha * sasrec_score + beta * llm_similarity

Key challenges:
1. Sequence alignment: LLM uses original item_ids, SASRec uses internal numeric IDs
2. Score calibration: Normalize both scores to [0,1] range
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
REVIEWS_PATH = os.path.join(SCRIPT_DIR, "../../Data/Amazons/data/All_Beauty.jsonl")
METADATA_PATH = os.path.join(SCRIPT_DIR, "../../Data/Amazons/data/meta_All_Beauty.jsonl")
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
MIN_SEQ_LENGTH = 3
MIN_ITEM_FREQ = 5
BATCH_SIZE = 8  # Smaller batch due to loading both models

# Fusion parameters (will be tuned)
FUSION_WEIGHTS = [0.3, 0.5, 0.7]  # Alpha values for SASRec weight


# =============================================================================
# Data Loading
# =============================================================================
def load_item_titles():
    """Load condensed item titles."""
    print("Loading item titles...")
    with open(TITLES_PATH, 'r', encoding='utf-8') as f:
        titles_data = json.load(f)

    # Create lookup: item_id -> condensed_title
    titles = {}
    for item in titles_data:
        titles[item['item_id']] = item['condensed_title']

    print(f"  Loaded {len(titles)} item titles")
    return titles


def load_item_mapping():
    """Load item_id <-> internal_id mapping from SASRec preprocessing."""
    print("Loading item mapping...")
    with open(ITEM_MAPPING_PATH, 'r') as f:
        mapping_data = json.load(f)

    item_to_id = mapping_data['item_to_id']  # original_id -> internal_id
    id_to_item = {v: k for k, v in item_to_id.items()}  # internal_id -> original_id
    num_items = mapping_data['num_items']

    print(f"  Loaded mapping for {num_items} items")
    return item_to_id, id_to_item, num_items


def load_llm_test_data():
    """Load LLM test data."""
    print("Loading LLM test data...")
    with open(LLM_TEST_DATA, 'r') as f:
        test_data = json.load(f)
    print(f"  Loaded {len(test_data)} test samples")
    return test_data


def load_sasrec_test_data():
    """Load SASRec test data for sequence alignment verification."""
    print("Loading SASRec test data...")
    with open(os.path.join(SASREC_DATA_PATH, "test.json"), 'r') as f:
        sasrec_test = json.load(f)
    print(f"  Loaded {len(sasrec_test)} SASRec test samples")
    return sasrec_test


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

    # Get vocab_size from model config
    vocab_size = model.config.vocab_size

    print(f"  SASRec model loaded: {model.num_parameters() / 1e6:.2f}M parameters")
    print(f"  Vocab size: {vocab_size}")
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

            # Mean pooling
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
def get_sasrec_scores_for_sequence(model, tokenizer, item_sequence, vocab_size, top_k=100, max_seq_length=100):
    """
    Get SASRec scores for a sequence of items.

    Args:
        model: SASRec model
        tokenizer: SASRec tokenizer
        item_sequence: List of internal item IDs (integers)
        vocab_size: Total vocab size (special tokens + items)
        top_k: Number of top items to return
        max_seq_length: Maximum sequence length (truncate if needed)

    Returns:
        Dictionary mapping item_id -> sasrec_score (normalized)
    """
    # Truncate sequence if too long
    if len(item_sequence) > max_seq_length:
        item_sequence = item_sequence[-max_seq_length:]

    # Tokenize sequence
    seq_str = " ".join(str(item_id) for item_id in item_sequence)
    inputs = tokenizer(seq_str, return_tensors='pt')
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last position logits

    # Get scores for all items (exclude special tokens 0-3)
    # vocab_size = 4 (special) + num_items
    # item token IDs: 4, 5, 6, ..., vocab_size-1
    # item internal IDs: 0, 1, 2, ..., num_items-1
    num_items = vocab_size - 4
    item_logits = logits[4:]  # Items start from token 4

    # Convert to probabilities via softmax
    probs = F.softmax(item_logits, dim=0)

    # Get top-k
    top_k_scores, top_k_indices = torch.topk(probs, min(top_k, num_items))

    # Create score dictionary (normalized to [0,1])
    scores = {}
    for idx, score in zip(top_k_indices.tolist(), top_k_scores.tolist()):
        scores[idx] = score  # item_internal_id -> score

    return scores, probs  # Keep on GPU for fusion


# =============================================================================
# Score Fusion
# =============================================================================
def fuse_scores(sasrec_probs, llm_similarity, alpha=0.5):
    """
    Fuse SASRec and LLM scores.

    Args:
        sasrec_probs: Tensor of SASRec probabilities [num_items]
        llm_similarity: Tensor of LLM cosine similarities [num_items]
        alpha: Weight for SASRec (1-alpha for LLM)

    Returns:
        Fused scores tensor
    """
    # Normalize LLM similarity to [0, 1]
    llm_min = llm_similarity.min()
    llm_max = llm_similarity.max()
    if llm_max - llm_min > 1e-6:
        llm_normalized = (llm_similarity - llm_min) / (llm_max - llm_min)
    else:
        llm_normalized = torch.ones_like(llm_similarity) * 0.5

    # Fuse
    fused = alpha * sasrec_probs + (1 - alpha) * llm_normalized

    return fused


def find_top_k_fused(fused_scores, item_list, k=20):
    """Find top-k items from fused scores."""
    top_k_values, top_k_indices = torch.topk(fused_scores, k)

    results = []
    for idx, score in zip(top_k_indices.tolist(), top_k_values.tolist()):
        results.append((item_list[idx], score))

    return results


# =============================================================================
# Main Evaluation
# =============================================================================
def main():
    print("=" * 60)
    print("Approach 2: Score Fusion (SASRec + LLM)")
    print("=" * 60)

    # Step 1: Load all data
    item_titles = load_item_titles()
    item_to_id, id_to_item, num_items = load_item_mapping()
    llm_test_data = load_llm_test_data()
    sasrec_test_data = load_sasrec_test_data()

    # Step 2: Create alignment verification
    # Build mapping: original_item_id -> condensed_title
    # and verify test data alignment

    print("\n" + "=" * 60)
    print("Verifying data alignment...")
    print("=" * 60)

    # Create a lookup from LLM test data
    llm_test_lookup = {}
    for sample in llm_test_data:
        key = (sample['user_id'], sample['target_item_id'])
        llm_test_lookup[key] = sample

    # Create a lookup from SASRec test data
    sasrec_test_lookup = {}
    for sample in sasrec_test_data:
        # ground_truth is internal item ID as string
        key = (sample['user_id'], sample['ground_truth'])
        sasrec_test_lookup[key] = sample

    # Verify alignment: for each LLM sample, find corresponding SASRec sample
    aligned_samples = []
    alignment_issues = 0

    for llm_sample in llm_test_data:
        user_id = llm_sample['user_id']
        target_item_id = llm_sample['target_item_id']  # Original ID like "B00XXX"

        # Convert to internal ID
        if target_item_id not in item_to_id:
            alignment_issues += 1
            continue

        target_internal_id = str(item_to_id[target_item_id])

        # Find SASRec sample
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
        else:
            alignment_issues += 1

    print(f"  LLM test samples: {len(llm_test_data)}")
    print(f"  SASRec test samples: {len(sasrec_test_data)}")
    print(f"  Aligned samples: {len(aligned_samples)}")
    print(f"  Alignment issues: {alignment_issues}")

    if len(aligned_samples) == 0:
        print("ERROR: No aligned samples found!")
        return

    # Step 3: Load models
    print("\n" + "=" * 60)
    print("Loading models...")
    print("=" * 60)

    sasrec_model, sasrec_tokenizer, sasrec_vocab_size = load_sasrec_model()
    llm_model, llm_tokenizer = load_llm_model()

    # Step 4: Pre-compute item embeddings for LLM
    # Get valid items (those in item_to_id mapping)
    valid_items = set(item_to_id.keys())
    item_embeddings, item_to_emb_idx, item_emb_list = get_title_embeddings(
        llm_model, llm_tokenizer, valid_items, item_titles, batch_size=64
    )

    print(f"  Item embeddings shape: {item_embeddings.shape}")

    # Step 5: Evaluate with different fusion weights
    print("\n" + "=" * 60)
    print("Evaluating Score Fusion...")
    print("=" * 60)

    results_by_alpha = {}

    for alpha in FUSION_WEIGHTS:
        print(f"\n--- Alpha = {alpha} (SASRec weight) ---")

        hr_scores = {k: [] for k in K_VALUES}
        ndcg_scores = {k: [] for k in K_VALUES}

        for aligned in tqdm(aligned_samples, desc=f"Evaluating (alpha={alpha})"):
            llm_sample = aligned['llm_sample']
            sasrec_sample = aligned['sasrec_sample']
            target_item_id = aligned['target_item_id']

            # 1. Get SASRec scores
            # Parse SASRec prompt (space-separated internal IDs)
            sasrec_prompt = sasrec_sample['prompt']
            sasrec_seq = [int(x) for x in sasrec_prompt.split()]

            sasrec_scores_dict, sasrec_probs = get_sasrec_scores_for_sequence(
                sasrec_model, sasrec_tokenizer, sasrec_seq, sasrec_vocab_size, top_k=sasrec_vocab_size
            )

            # 2. Get LLM prediction embedding
            # Generate prediction text
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

            # Get prediction embedding
            pred_embedding = get_prediction_embedding(llm_model, llm_tokenizer, predicted_text)

            # 3. Compute LLM similarity with all items
            pred_norm = F.normalize(pred_embedding.unsqueeze(0), dim=1)
            item_norms = F.normalize(item_embeddings, dim=1)
            llm_similarities = torch.mm(pred_norm, item_norms.T).squeeze(0)

            # 4. Fuse scores
            # Map item_emb_list indices to internal item IDs
            # item_emb_list[i] is original_id, we need to convert to internal_id
            # But sasrec_probs is indexed by internal_id (0 to num_items-1)

            # Create a mapping from embedding index to internal ID
            emb_idx_to_internal = {}
            for emb_idx, orig_id in enumerate(item_emb_list):
                if orig_id in item_to_id:
                    emb_idx_to_internal[emb_idx] = item_to_id[orig_id]

            # Build fused scores array indexed by embedding index
            fused_scores_list = []
            for emb_idx in range(len(item_emb_list)):
                internal_id = emb_idx_to_internal.get(emb_idx)
                if internal_id is not None and internal_id < len(sasrec_probs):
                    sasrec_score = sasrec_probs[internal_id]
                    llm_score = llm_similarities[emb_idx]
                    fused = alpha * sasrec_score + (1 - alpha) * llm_score
                    fused_scores_list.append(fused)
                else:
                    # Fallback to LLM score only
                    fused_scores_list.append((1 - alpha) * llm_similarities[emb_idx])

            fused_scores = torch.stack(fused_scores_list)

            # 5. Get top-k from fused scores
            top_k_items = find_top_k_fused(fused_scores, item_emb_list, k=max(K_VALUES))

            # 6. Compute metrics
            for k in K_VALUES:
                top_k_ids = [item[0] for item in top_k_items[:k]]
                if target_item_id in top_k_ids:
                    hr_scores[k].append(1.0)
                    rank = top_k_ids.index(target_item_id) + 1
                    ndcg_scores[k].append(1.0 / np.log2(rank + 1))
                else:
                    hr_scores[k].append(0.0)
                    ndcg_scores[k].append(0.0)

        # Compute average metrics
        results = {
            'alpha': alpha,
            'HR': {f'HR@{k}': np.mean(scores) for k, scores in hr_scores.items()},
            'NDCG': {f'NDCG@{k}': np.mean(scores) for k, scores in ndcg_scores.items()}
        }
        results_by_alpha[alpha] = results

        print(f"\nResults (alpha={alpha}):")
        for k in K_VALUES:
            print(f"  HR@{k}: {results['HR'][f'HR@{k}']:.4f}")
            print(f"  NDCG@{k}: {results['NDCG'][f'NDCG@{k}']:.4f}")

    # Step 6: Print final comparison
    print("\n" + "=" * 60)
    print("Final Results Comparison")
    print("=" * 60)

    print("\nBaseline:")
    print("  SASRec alone:    HR@1=9.22%, HR@10=13.03%")
    print("  LLM alone:       HR@1=9.26%, HR@10=12.18%")

    print("\nFusion Results:")
    for alpha, results in sorted(results_by_alpha.items()):
        print(f"\n  Alpha={alpha}:")
        for k in [1, 10]:
            print(f"    HR@{k}: {results['HR'][f'HR@{k}']:.4f} ({results['HR'][f'HR@{k}']*100:.2f}%)")

    # Step 7: Save results
    output_path = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_path, exist_ok=True)

    final_results = {
        'fusion_weights': FUSION_WEIGHTS,
        'results_by_alpha': {str(k): v for k, v in results_by_alpha.items()},
        'num_test_samples': len(aligned_samples),
        'baselines': {
            'sasrec': {'HR@1': 0.0922, 'HR@10': 0.1303},
            'llm': {'HR@1': 0.0926, 'HR@10': 0.1218}
        }
    }

    with open(os.path.join(output_path, "score_fusion_results.json"), 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to {output_path}/score_fusion_results.json")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
