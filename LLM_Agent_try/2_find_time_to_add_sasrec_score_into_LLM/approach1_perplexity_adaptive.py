#!/usr/bin/env python3
"""
Approach 1: Perplexity-based Adaptive Fusion

Key idea: Use LLM's perplexity during generation to determine fusion weight.
- High perplexity = LLM is uncertain -> rely more on SASRec
- Low perplexity = LLM is confident -> rely more on LLM

Adaptive formula:
  alpha_adaptive = alpha_base + k * (perplexity - threshold) / scale

Where:
  - alpha_base: base SASRec weight (e.g., 0.5)
  - k: sensitivity factor
  - threshold: perplexity threshold (median or mean)
  - scale: normalization factor (std of perplexity)
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

# Data paths (relative to this script's location)
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

# Adaptive fusion parameters
BASE_ALPHA = 0.5  # Base SASRec weight
PERPLEXITY_SENSITIVITY = 0.1  # How much to adjust based on perplexity


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
# Perplexity Computation
# =============================================================================
def compute_perplexity(model, tokenizer, prompt, generated_text):
    """
    Compute perplexity of the generated text.

    Perplexity = exp(average_nll)
    Lower perplexity = more confident prediction
    """
    # Combine prompt and generated text
    full_text = prompt + generated_text

    inputs = tokenizer(full_text, return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        # Loss is the average negative log likelihood
        loss = outputs.loss

    # Perplexity = exp(loss)
    perplexity = torch.exp(loss).item()

    return perplexity


def compute_generation_perplexity(model, tokenizer, prompt):
    """
    Compute perplexity during generation.
    We generate first, then compute perplexity of the generation.
    """
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

    with torch.no_grad():
        # Generate prediction
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    input_len = inputs['input_ids'].shape[1]
    generated_ids = outputs.sequences[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Compute perplexity of generated tokens
    # We need to compute the loss for just the generated part
    if len(generated_ids) == 0:
        perplexity = 100.0  # High uncertainty if no generation
    else:
        # Get logits for generated tokens
        scores = outputs.scores  # List of tensors, one per generated token

        nll_sum = 0.0
        for i, score in enumerate(scores):
            if i >= len(generated_ids):
                break
            log_probs = F.log_softmax(score[0], dim=-1)
            token_id = generated_ids[i]
            nll_sum += -log_probs[token_id].item()

        avg_nll = nll_sum / max(len(generated_ids), 1)
        perplexity = np.exp(avg_nll)

    return generated_text, perplexity


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
# Adaptive Fusion
# =============================================================================
def adaptive_fuse_scores(sasrec_probs, llm_similarity, alpha_base, perplexity,
                         perplexity_median, perplexity_std, sensitivity=0.1):
    """
    Adaptively fuse scores based on perplexity.

    High perplexity -> more SASRec weight
    Low perplexity -> more LLM weight
    """
    # Normalize perplexity
    normalized_ppl = (perplexity - perplexity_median) / (perplexity_std + 1e-6)

    # Adjust alpha: positive normalized_ppl means high perplexity -> increase alpha
    alpha_adaptive = alpha_base + sensitivity * normalized_ppl

    # Clamp to [0.1, 0.9] to avoid extreme weights
    alpha_adaptive = max(0.1, min(0.9, alpha_adaptive))

    # Normalize LLM similarity
    llm_min = llm_similarity.min()
    llm_max = llm_similarity.max()
    if llm_max - llm_min > 1e-6:
        llm_normalized = (llm_similarity - llm_min) / (llm_max - llm_min)
    else:
        llm_normalized = torch.ones_like(llm_similarity) * 0.5

    # Fuse
    fused = alpha_adaptive * sasrec_probs + (1 - alpha_adaptive) * llm_normalized

    return fused, alpha_adaptive


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
    print("Approach 1: Perplexity-based Adaptive Fusion")
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
    global llm_model
    sasrec_model, sasrec_tokenizer, sasrec_vocab_size = load_sasrec_model()
    llm_model, llm_tokenizer = load_llm_model()

    # Step 4: Pre-compute item embeddings
    valid_items = set(item_to_id.keys())
    item_embeddings, item_to_emb_idx, item_emb_list = get_title_embeddings(
        llm_model, llm_tokenizer, valid_items, item_titles, batch_size=64
    )

    print(f"  Item embeddings shape: {item_embeddings.shape}")

    # Step 5: First pass - compute all perplexities
    print("\n" + "=" * 60)
    print("Phase 1: Computing perplexities for all samples...")
    print("=" * 60)

    perplexities = []
    generation_results = []

    for aligned in tqdm(aligned_samples, desc="Computing perplexities"):
        llm_sample = aligned['llm_sample']
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
                output_scores=True,
                return_dict_in_generate=True,
            )

        input_len = inputs['input_ids'].shape[1]
        generated_ids = outputs.sequences[0][input_len:]
        generated_text = llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Compute perplexity
        if len(generated_ids) == 0:
            perplexity = 100.0
        else:
            scores = outputs.scores
            nll_sum = 0.0
            for i, score in enumerate(scores):
                if i >= len(generated_ids):
                    break
                log_probs = F.log_softmax(score[0], dim=-1)
                token_id = generated_ids[i]
                nll_sum += -log_probs[token_id].item()

            avg_nll = nll_sum / max(len(generated_ids), 1)
            perplexity = np.exp(avg_nll)

        perplexities.append(perplexity)
        generation_results.append({
            'generated_text': generated_text,
            'perplexity': perplexity
        })

    # Compute perplexity statistics
    ppl_median = np.median(perplexities)
    ppl_mean = np.mean(perplexities)
    ppl_std = np.std(perplexities)

    print(f"\nPerplexity Statistics:")
    print(f"  Mean: {ppl_mean:.2f}")
    print(f"  Median: {ppl_median:.2f}")
    print(f"  Std: {ppl_std:.2f}")
    print(f"  Min: {min(perplexities):.2f}")
    print(f"  Max: {max(perplexities):.2f}")

    # Step 6: Evaluate with different sensitivity values
    print("\n" + "=" * 60)
    print("Phase 2: Evaluating Adaptive Fusion...")
    print("=" * 60)

    sensitivities = [0.05, 0.1, 0.2]  # How much to adjust based on perplexity
    base_alphas = [0.5, 0.7]  # Base fusion weights

    all_results = {}

    for base_alpha in base_alphas:
        for sensitivity in sensitivities:
            config_name = f"alpha_{base_alpha}_sens_{sensitivity}"
            print(f"\n--- Config: base_alpha={base_alpha}, sensitivity={sensitivity} ---")

            hr_scores = {k: [] for k in K_VALUES}
            ndcg_scores = {k: [] for k in K_VALUES}
            alpha_values_used = []

            for i, aligned in enumerate(tqdm(aligned_samples, desc=f"Eval {config_name}")):
                sasrec_sample = aligned['sasrec_sample']
                target_item_id = aligned['target_item_id']
                gen_result = generation_results[i]

                generated_text = gen_result['generated_text']
                perplexity = gen_result['perplexity']

                # 1. Get SASRec scores
                sasrec_prompt = sasrec_sample['prompt']
                sasrec_seq = [int(x) for x in sasrec_prompt.split()]
                sasrec_probs = get_sasrec_scores_for_sequence(
                    sasrec_model, sasrec_tokenizer, sasrec_seq, sasrec_vocab_size
                )

                # 2. Get LLM prediction embedding
                pred_embedding = get_prediction_embedding(llm_model, llm_tokenizer, generated_text)

                # 3. Compute LLM similarity
                pred_norm = F.normalize(pred_embedding.unsqueeze(0), dim=1)
                item_norms = F.normalize(item_embeddings, dim=1)
                llm_similarities = torch.mm(pred_norm, item_norms.T).squeeze(0)

                # 4. Adaptive fusion
                emb_idx_to_internal = {}
                for emb_idx, orig_id in enumerate(item_emb_list):
                    if orig_id in item_to_id:
                        emb_idx_to_internal[emb_idx] = item_to_id[orig_id]

                fused_scores_list = []
                for emb_idx in range(len(item_emb_list)):
                    internal_id = emb_idx_to_internal.get(emb_idx)
                    if internal_id is not None and internal_id < len(sasrec_probs):
                        sasrec_score = sasrec_probs[internal_id]
                        llm_score = llm_similarities[emb_idx]
                        fused_scores_list.append((sasrec_score, llm_score))

                # Apply adaptive fusion
                if fused_scores_list:
                    sasrec_t = torch.stack([f[0] for f in fused_scores_list])
                    llm_t = torch.stack([f[1] for f in fused_scores_list])

                    fused, alpha_used = adaptive_fuse_scores(
                        sasrec_t, llm_t, base_alpha, perplexity,
                        ppl_median, ppl_std, sensitivity
                    )

                    alpha_values_used.append(alpha_used)

                    # Get top-k
                    top_k_items = find_top_k_fused(fused,
                        [item_emb_list[i] for i in range(len(fused_scores_list))],
                        k=max(K_VALUES))

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
                'base_alpha': base_alpha,
                'sensitivity': sensitivity,
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
    print("  SASRec alone:       HR@1=9.22%, HR@10=13.03%")
    print("  LLM alone:          HR@1=9.26%, HR@10=12.18%")
    print("  Score Fusion (α=0.7): HR@1=10.95%, HR@10=13.25%")

    print("\nAdaptive Fusion Results:")
    for config_name, results in sorted(all_results.items()):
        print(f"\n  {config_name}:")
        print(f"    HR@1: {results['HR']['HR@1']*100:.2f}%")
        print(f"    HR@10: {results['HR']['HR@10']*100:.2f}%")
        print(f"    Avg alpha: {results['avg_alpha_used']:.3f}")

    # Step 8: Save results
    output_path = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_path, exist_ok=True)

    final_results = {
        'approach': 'perplexity_adaptive',
        'perplexity_stats': {
            'mean': float(ppl_mean),
            'median': float(ppl_median),
            'std': float(ppl_std),
            'min': float(min(perplexities)),
            'max': float(max(perplexities))
        },
        'results_by_config': all_results,
        'num_test_samples': len(aligned_samples),
        'baselines': {
            'sasrec': {'HR@1': 0.0922, 'HR@10': 0.1303},
            'llm': {'HR@1': 0.0926, 'HR@10': 0.1218},
            'score_fusion_alpha_0.7': {'HR@1': 0.1095, 'HR@10': 0.1325}
        }
    }

    with open(os.path.join(output_path, "perplexity_adaptive_results.json"), 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to {output_path}/perplexity_adaptive_results.json")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
