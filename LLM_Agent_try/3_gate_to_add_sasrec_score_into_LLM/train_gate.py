#!/usr/bin/env python3
"""
Gate Network Training: Learn when to use SASRec score

Three approaches:
- A (Shallow): Gate from Qwen shallow layer (layer 8)
- B (Deep): Gate from Qwen final layer (layer -1)
- C (Hybrid): Gate from both shallow and deep layers

Training strategy:
- Freeze LLM and SASRec
- Only train the Gate MLP
- Loss: 1 - HR@1 after fusion
- Early stopping on validation set
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
from collections import defaultdict
import gc

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
RAW_DATA_PATH = os.path.join(SCRIPT_DIR, "../../Data/Amazons/data/All_Beauty.jsonl")

# SASRec paths
SASREC_CHECKPOINT = os.path.join(SCRIPT_DIR, "../vanilla_sasrec/checkpoints/sasrec_beauty_20260226_055626/best_model")
ITEM_MAPPING_PATH = os.path.join(SCRIPT_DIR, "../vanilla_sasrec/processed_data/item_mapping.json")

# LLM path
LLM_MODEL_PATH = os.path.join(SCRIPT_DIR, "../../LLM4RecPart/models/Qwen3-1-7B")

# Training settings
BATCH_SIZE = 64  # Will be tuned based on GPU memory
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
HIDDEN_DIM = 512

# Gate types
GATE_TYPES = ['shallow', 'deep', 'hybrid']

# Shallow layer index (Qwen has 28 layers, use layer 8 as shallow)
SHALLOW_LAYER_IDX = 8

# GPU
DEVICE = "cuda:0"


# =============================================================================
# Gate Network
# =============================================================================
class GateNetwork(nn.Module):
    """
    MLP that outputs a gate score to decide how much to use SASRec score.
    Output is in [0, 1] via sigmoid.
    """
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)  # [batch_size]


# =============================================================================
# Data Loading and Preparation
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


def load_and_split_data(item_to_id):
    """
    Load raw data and split into train/val/test.
    Split strategy:
    - For each user sequence: [items[:-2]] -> train, [items[-2]] -> val, [items[-1]] -> test
    """
    print("Loading and splitting data...")

    # Load raw interactions
    user_sequences = defaultdict(list)
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            user_id = data.get('reviewerID', data.get('user_id'))
            item_id = data.get('asin', data.get('item_id'))
            timestamp = data.get('unixReviewTime', data.get('timestamp', 0))
            rating = data.get('overall', data.get('rating', 5))

            if item_id in item_to_id:
                user_sequences[user_id].append({
                    'item_id': item_id,
                    'timestamp': timestamp,
                    'rating': rating
                })

    # Sort by timestamp and split
    train_data = []
    val_data = []
    test_data = []

    for user_id, items in user_sequences.items():
        if len(items) < 3:  # Need at least 3 items for split
            continue

        # Sort by timestamp
        items = sorted(items, key=lambda x: x['timestamp'])

        # Split: train=all except last 2, val=second to last, test=last
        for i in range(2, len(items)):
            history = items[:i]
            target = items[i]

            sample = {
                'user_id': user_id,
                'history': history,
                'target_item_id': target['item_id'],
                'target_rating': target['rating']
            }

            if i == len(items) - 1:
                test_data.append(sample)
            elif i == len(items) - 2:
                val_data.append(sample)
            else:
                train_data.append(sample)

    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Test samples: {len(test_data)}")

    return train_data, val_data, test_data


# =============================================================================
# Model Loading
# =============================================================================
def load_sasrec_model():
    """Load SASRec model and tokenizer."""
    print(f"Loading SASRec model...")

    model = SasRecForCausalLM.from_pretrained(SASREC_CHECKPOINT)
    model = model.to(DEVICE)
    model.eval()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(SASREC_CHECKPOINT)
    tokenizer.padding_side = 'left'

    vocab_size = model.config.vocab_size

    print(f"  SASRec model loaded: {model.num_parameters() / 1e6:.2f}M parameters")
    return model, tokenizer, vocab_size


def load_llm_model():
    """Load LLM model and tokenizer."""
    print(f"Loading LLM model...")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        output_hidden_states=True,  # Need hidden states for gate
    ).to(DEVICE)
    model.eval()

    print(f"  LLM model loaded")
    return model, tokenizer


# =============================================================================
# Embedding Pre-computation
# =============================================================================
def get_title_embeddings(model, tokenizer, item_ids, item_titles, batch_size=128):
    """Pre-compute embeddings for all item titles."""
    print(f"\nPre-computing embeddings for {len(item_ids)} items...")

    embeddings = []
    item_list = list(item_ids)

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(item_list), batch_size), desc="Encoding titles"):
            batch_items = item_list[i:i+batch_size]
            batch_titles = [item_titles.get(item, "Unknown") for item in batch_items]

            inputs = tokenizer(batch_titles, return_tensors='pt', padding=True, truncation=True, max_length=50)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model.model(**inputs, use_cache=False)
            hidden_states = outputs.last_hidden_state

            # Mean pooling
            attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
            masked_hidden = hidden_states * attention_mask
            batch_embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)

            embeddings.append(batch_embeddings.cpu())

    all_embeddings = torch.cat(embeddings, dim=0)
    item_to_idx = {item: idx for idx, item in enumerate(item_list)}

    return all_embeddings.to(DEVICE), item_to_idx, item_list


# =============================================================================
# Scoring Functions
# =============================================================================
def get_sasrec_scores(model, tokenizer, item_sequences, vocab_size, batch_size=64):
    """Get SASRec scores for a batch of sequences."""
    all_probs = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(item_sequences), batch_size):
            batch_seqs = item_sequences[i:i+batch_size]

            # Tokenize
            seq_strs = [" ".join(str(idx) for idx in seq) for seq in batch_seqs]
            inputs = tokenizer(seq_strs, return_tensors='pt', padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # Last position

            # Get item probabilities (exclude special tokens 0-3)
            item_logits = logits[:, 4:]
            probs = F.softmax(item_logits, dim=-1)

            all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0).to(DEVICE)


def get_llm_hidden_states_and_predictions(model, tokenizer, prompts, batch_size=8, shallow_layer=8):
    """
    Get LLM hidden states and predictions for a batch of prompts.

    Returns:
        shallow_hidden: [batch_size, hidden_dim] from shallow layer
        deep_hidden: [batch_size, hidden_dim] from final layer
        predictions: list of predicted texts
    """
    all_shallow = []
    all_deep = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]

            inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # Generate predictions
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

            input_len = inputs['input_ids'].shape[1]

            # Get predictions
            for j, seq in enumerate(outputs.sequences):
                pred_text = tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
                all_predictions.append(pred_text)

            # Also run forward pass to get hidden states of the prompt
            forward_outputs = model(**inputs, output_hidden_states=True)

            # Get hidden states: [batch, seq_len, hidden_dim]
            hidden_states = forward_outputs.hidden_states

            # Shallow layer (layer 8)
            shallow_h = hidden_states[shallow_layer]  # [batch, seq_len, hidden]
            shallow_h = shallow_h[:, -1, :]  # Last token: [batch, hidden]
            all_shallow.append(shallow_h.cpu())

            # Deep layer (last layer)
            deep_h = hidden_states[-1]  # [batch, seq_len, hidden]
            deep_h = deep_h[:, -1, :]  # Last token: [batch, hidden]
            all_deep.append(deep_h.cpu())

    shallow_hidden = torch.cat(all_shallow, dim=0).to(DEVICE)
    deep_hidden = torch.cat(all_deep, dim=0).to(DEVICE)

    return shallow_hidden, deep_hidden, all_predictions


def get_prediction_embedding(model, tokenizer, predictions, batch_size=64):
    """Get embeddings for predicted texts."""
    embeddings = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(predictions), batch_size):
            batch_preds = predictions[i:i+batch_size]

            inputs = tokenizer(batch_preds, return_tensors='pt', padding=True, truncation=True, max_length=50)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model.model(**inputs, use_cache=False)
            hidden_states = outputs.last_hidden_state

            attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
            masked_hidden = hidden_states * attention_mask
            batch_embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)

            embeddings.append(batch_embeddings.cpu())

    return torch.cat(embeddings, dim=0).to(DEVICE)


# =============================================================================
# Training Functions
# =============================================================================
def compute_fused_scores(sasrec_probs, llm_similarities, gate_scores, item_to_emb_idx, item_to_id, item_emb_list):
    """
    Compute fused scores using gate.

    fused = gate * sasrec_prob + (1 - gate) * llm_similarity
    """
    batch_size = len(gate_scores)
    num_items = len(item_emb_list)

    # Normalize LLM similarities
    llm_min = llm_similarities.min(dim=1, keepdim=True)[0]
    llm_max = llm_similarities.max(dim=1, keepdim=True)[0]
    llm_range = llm_max - llm_min
    llm_range = torch.where(llm_range < 1e-6, torch.ones_like(llm_range), llm_range)
    llm_normalized = (llm_similarities - llm_min) / llm_range

    # Fuse scores
    gate_scores = gate_scores.unsqueeze(1)  # [batch, 1]
    fused = gate_scores * sasrec_probs + (1 - gate_scores) * llm_normalized

    return fused


def compute_hr_at_k(fused_scores, target_item_ids, item_emb_list, item_to_id, k=1):
    """Compute HR@K for a batch."""
    batch_size = fused_scores.shape[0]
    hits = []

    for i in range(batch_size):
        scores = fused_scores[i]
        top_k_indices = torch.topk(scores, k).indices.tolist()
        top_k_items = [item_emb_list[idx] for idx in top_k_indices]

        target_id = target_item_ids[i]
        hit = 1.0 if target_id in top_k_items else 0.0
        hits.append(hit)

    return torch.tensor(hits, device=DEVICE)


def train_epoch(gate_model, optimizer, train_data, sasrec_model, sasrec_tokenizer,
                llm_model, llm_tokenizer, item_embeddings, item_to_emb_idx,
                item_emb_list, item_to_id, vocab_size, batch_size, gate_type='shallow'):
    """Train for one epoch."""
    gate_model.train()

    # Shuffle data
    indices = np.random.permutation(len(train_data))

    total_loss = 0
    total_hr = 0
    num_batches = 0

    for start_idx in range(0, len(train_data), batch_size):
        batch_indices = indices[start_idx:start_idx+batch_size]
        batch_data = [train_data[i] for i in batch_indices]

        # Prepare batch
        item_seqs = []
        prompts = []
        target_ids = []

        for sample in batch_data:
            # SASRec sequence (internal IDs)
            seq = [item_to_id[item['item_id']] for item in sample['history']]
            item_seqs.append(seq)

            # LLM prompt
            history_str = "\n".join([
                f'{i+1}. "{sample["history"][i].get("title", "Product")}" (Rating: {sample["history"][i]["rating"]})'
                for i in range(min(len(sample["history"]), 10))
            ])
            prompt = f"User's purchase history:\n{history_str}\n\nNext product:"
            prompts.append(prompt)

            target_ids.append(sample['target_item_id'])

        # Get SASRec scores
        sasrec_probs = get_sasrec_scores(sasrec_model, sasrec_tokenizer, item_seqs, vocab_size, batch_size=min(batch_size, 16))

        # Get LLM hidden states and predictions
        shallow_h, deep_h, predictions = get_llm_hidden_states_and_predictions(
            llm_model, llm_tokenizer, prompts, batch_size=min(batch_size, 8),
            shallow_layer=SHALLOW_LAYER_IDX
        )

        # Get prediction embeddings
        pred_embeddings = get_prediction_embedding(llm_model, llm_tokenizer, predictions, batch_size=min(batch_size, 32))

        # Compute LLM similarities
        pred_norm = F.normalize(pred_embeddings, dim=1)
        item_norms = F.normalize(item_embeddings, dim=1)
        llm_similarities = torch.mm(pred_norm, item_norms.T)  # [batch, num_items]

        # Select hidden states based on gate type
        if gate_type == 'shallow':
            hidden_states = shallow_h.float()
        elif gate_type == 'deep':
            hidden_states = deep_h.float()
        else:  # hybrid
            hidden_states = torch.cat([shallow_h, deep_h], dim=1).float()

        # Get gate scores
        gate_scores = gate_model(hidden_states)

        # Compute fused scores
        fused_scores = compute_fused_scores(
            sasrec_probs, llm_similarities, gate_scores,
            item_to_emb_idx, item_to_id, item_emb_list
        )

        # Compute HR@1 as reward
        hr1 = compute_hr_at_k(fused_scores, target_ids, item_emb_list, item_to_id, k=1)

        # Loss: 1 - HR@1 (we want to maximize HR)
        loss = 1 - hr1.mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_hr += hr1.mean().item()
        num_batches += 1

        # Clear cache
        if num_batches % 10 == 0:
            torch.cuda.empty_cache()

    return total_loss / num_batches, total_hr / num_batches


def evaluate(gate_model, eval_data, sasrec_model, sasrec_tokenizer,
             llm_model, llm_tokenizer, item_embeddings, item_to_emb_idx,
             item_emb_list, item_to_id, vocab_size, batch_size, gate_type='shallow'):
    """Evaluate on validation/test set."""
    gate_model.eval()

    all_hr1 = []
    all_hr10 = []

    with torch.no_grad():
        for start_idx in range(0, len(eval_data), batch_size):
            batch_data = eval_data[start_idx:start_idx+batch_size]

            # Prepare batch
            item_seqs = []
            prompts = []
            target_ids = []

            for sample in batch_data:
                seq = [item_to_id[item['item_id']] for item in sample['history']]
                item_seqs.append(seq)

                history_str = "\n".join([
                    f'{i+1}. "Product" (Rating: {sample["history"][i]["rating"]})'
                    for i in range(min(len(sample["history"]), 10))
                ])
                prompt = f"User's purchase history:\n{history_str}\n\nNext product:"
                prompts.append(prompt)

                target_ids.append(sample['target_item_id'])

            # Get SASRec scores
            sasrec_probs = get_sasrec_scores(sasrec_model, sasrec_tokenizer, item_seqs, vocab_size, batch_size=min(batch_size, 16))

            # Get LLM hidden states and predictions
            shallow_h, deep_h, predictions = get_llm_hidden_states_and_predictions(
                llm_model, llm_tokenizer, prompts, batch_size=min(batch_size, 8),
                shallow_layer=SHALLOW_LAYER_IDX
            )

            # Get prediction embeddings
            pred_embeddings = get_prediction_embedding(llm_model, llm_tokenizer, predictions, batch_size=min(batch_size, 32))

            # Compute LLM similarities
            pred_norm = F.normalize(pred_embeddings, dim=1)
            item_norms = F.normalize(item_embeddings, dim=1)
            llm_similarities = torch.mm(pred_norm, item_norms.T)

            # Select hidden states based on gate type
            if gate_type == 'shallow':
                hidden_states = shallow_h.float()
            elif gate_type == 'deep':
                hidden_states = deep_h.float()
            else:
                hidden_states = torch.cat([shallow_h, deep_h], dim=1).float()

            # Get gate scores
            gate_scores = gate_model(hidden_states)

            # Compute fused scores
            fused_scores = compute_fused_scores(
                sasrec_probs, llm_similarities, gate_scores,
                item_to_emb_idx, item_to_id, item_emb_list
            )

            # Compute HR@1 and HR@10
            hr1 = compute_hr_at_k(fused_scores, target_ids, item_emb_list, item_to_id, k=1)
            hr10 = compute_hr_at_k(fused_scores, target_ids, item_emb_list, item_to_id, k=10)

            all_hr1.extend(hr1.tolist())
            all_hr10.extend(hr10.tolist())

    return np.mean(all_hr1), np.mean(all_hr10)


# =============================================================================
# Main Training Loop
# =============================================================================
def train_gate(gate_type, train_data, val_data, test_data,
               sasrec_model, sasrec_tokenizer, vocab_size,
               llm_model, llm_tokenizer, item_embeddings, item_to_emb_idx,
               item_emb_list, item_to_id, num_items):
    """Train gate network with specified type."""

    print(f"\n{'='*60}")
    print(f"Training Gate Network - Type: {gate_type}")
    print(f"{'='*60}")

    # Determine input dimension
    if gate_type == 'shallow':
        input_dim = 2048  # Qwen hidden size
    elif gate_type == 'deep':
        input_dim = 2048
    else:  # hybrid
        input_dim = 4096

    # Create model
    gate_model = GateNetwork(input_dim, HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(gate_model.parameters(), lr=LEARNING_RATE)

    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Parameters: {sum(p.numel() for p in gate_model.parameters()):,}")

    # Training loop
    best_val_hr1 = 0
    patience_counter = 0
    best_model_state = None

    results_history = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Train
        train_loss, train_hr1 = train_epoch(
            gate_model, optimizer, train_data,
            sasrec_model, sasrec_tokenizer,
            llm_model, llm_tokenizer,
            item_embeddings, item_to_emb_idx,
            item_emb_list, item_to_id,
            vocab_size, BATCH_SIZE, gate_type
        )

        # Validate
        val_hr1, val_hr10 = evaluate(
            gate_model, val_data,
            sasrec_model, sasrec_tokenizer,
            llm_model, llm_tokenizer,
            item_embeddings, item_to_emb_idx,
            item_emb_list, item_to_id,
            vocab_size, BATCH_SIZE, gate_type
        )

        print(f"  Train Loss: {train_loss:.4f}, Train HR@1: {train_hr1:.4f}")
        print(f"  Val HR@1: {val_hr1:.4f}, Val HR@10: {val_hr10:.4f}")

        results_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_hr1': train_hr1,
            'val_hr1': val_hr1,
            'val_hr10': val_hr10
        })

        # Early stopping
        if val_hr1 > best_val_hr1:
            best_val_hr1 = val_hr1
            patience_counter = 0
            best_model_state = gate_model.state_dict().copy()
            print(f"  New best validation HR@1: {best_val_hr1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model and test
    if best_model_state is not None:
        gate_model.load_state_dict(best_model_state)

    test_hr1, test_hr10 = evaluate(
        gate_model, test_data,
        sasrec_model, sasrec_tokenizer,
        llm_model, llm_tokenizer,
        item_embeddings, item_to_emb_idx,
        item_emb_list, item_to_id,
        vocab_size, BATCH_SIZE, gate_type
    )

    print(f"\nFinal Test Results:")
    print(f"  Test HR@1: {test_hr1:.4f} ({test_hr1*100:.2f}%)")
    print(f"  Test HR@10: {test_hr10:.4f} ({test_hr10*100:.2f}%)")

    return {
        'gate_type': gate_type,
        'input_dim': input_dim,
        'best_val_hr1': best_val_hr1,
        'test_hr1': test_hr1,
        'test_hr10': test_hr10,
        'history': results_history
    }


# =============================================================================
# Main
# =============================================================================
def main():
    print("="*60)
    print("Gate Network Training for Adaptive Score Fusion")
    print("="*60)

    # Check GPU
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    item_titles = load_item_titles()
    item_to_id, id_to_item, num_items = load_item_mapping()
    train_data, val_data, test_data = load_and_split_data(item_to_id)

    # Load models
    sasrec_model, sasrec_tokenizer, vocab_size = load_sasrec_model()
    llm_model, llm_tokenizer = load_llm_model()

    # Pre-compute item embeddings
    valid_items = set(item_to_id.keys())
    item_embeddings, item_to_emb_idx, item_emb_list = get_title_embeddings(
        llm_model, llm_tokenizer, valid_items, item_titles, batch_size=128
    )

    print(f"\nItem embeddings shape: {item_embeddings.shape}")

    # Train all gate types
    all_results = {}

    for gate_type in GATE_TYPES:
        result = train_gate(
            gate_type, train_data, val_data, test_data,
            sasrec_model, sasrec_tokenizer, vocab_size,
            llm_model, llm_tokenizer,
            item_embeddings, item_to_emb_idx,
            item_emb_list, item_to_id, num_items
        )
        all_results[gate_type] = result

        # Save intermediate results
        output_path = os.path.join(SCRIPT_DIR, "results")
        os.makedirs(output_path, exist_ok=True)

        with open(os.path.join(output_path, f"gate_{gate_type}_results.json"), 'w') as f:
            json.dump(result, f, indent=2)

        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

    # Print final comparison
    print("\n" + "="*60)
    print("Final Results Comparison")
    print("="*60)

    print("\nBaselines:")
    print("  SASRec alone:       HR@1=9.22%, HR@10=13.03%")
    print("  LLM alone:          HR@1=9.26%, HR@10=12.18%")
    print("  Score Fusion (α=0.7): HR@1=10.95%, HR@10=13.25%")

    print("\nGate Network Results:")
    for gate_type, result in all_results.items():
        print(f"\n  {gate_type.upper()}:")
        print(f"    Val HR@1: {result['best_val_hr1']*100:.2f}%")
        print(f"    Test HR@1: {result['test_hr1']*100:.2f}%")
        print(f"    Test HR@10: {result['test_hr10']*100:.2f}%")

    # Save final results
    final_results = {
        'all_results': all_results,
        'baselines': {
            'sasrec': {'HR@1': 0.0922, 'HR@10': 0.1303},
            'llm': {'HR@1': 0.0926, 'HR@10': 0.1218},
            'score_fusion_alpha_0.7': {'HR@1': 0.1095, 'HR@10': 0.1325}
        }
    }

    with open(os.path.join(output_path, "gate_all_results.json"), 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}/")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
