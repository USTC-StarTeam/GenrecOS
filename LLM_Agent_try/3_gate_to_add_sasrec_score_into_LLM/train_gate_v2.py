#!/usr/bin/env python3
"""
Context-adaptive full-catalog fusion for SASRec + Qwen signals.

This replaces the old scalar gate design with a sample-conditioned fusion model:
1. Use real aligned LLM prompts for train / val / test.
2. Pre-compute full-catalog SASRec probabilities and LLM semantic scores in batches.
3. Train a context encoder on prompt hidden states + confidence statistics.
4. Predict sample-specific fusion coefficients over item-wise score channels.

The model keeps the strong fixed-fusion baseline as a residual anchor and learns
small context-aware corrections on top of it.
"""

import argparse
import gc
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

sys.path.append("../../Rec-Transformer")
sys.path.append("../..")
from sasrec import SasRecForCausalLM


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

TITLES_PATH = os.path.join(ROOT_DIR, "use_Qwen3-1-7B_to_generate_title", "item_titles_unique.json")
ITEM_MAPPING_PATH = os.path.join(ROOT_DIR, "vanilla_sasrec", "processed_data", "item_mapping.json")
SASREC_SPLIT_DIR = os.path.join(ROOT_DIR, "vanilla_sasrec", "processed_data", "splits")
SASREC_CHECKPOINT = os.path.join(
    ROOT_DIR,
    "vanilla_sasrec",
    "checkpoints",
    "sasrec_beauty_20260226_055626",
    "best_model",
)
LLM_DATA_DIR = os.path.join(ROOT_DIR, "LLM_Rec_Data_Preparation")
LLM_MODEL_PATH = os.path.join(ROOT_DIR, "..", "LLM4RecPart", "models", "Qwen3-1-7B")

CACHE_DIR = os.path.join(SCRIPT_DIR, "cache", "context_adaptive_v1")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

MAX_K = 20
SHALLOW_LAYER_IDX = 8
TITLE_MAX_LENGTH = 56


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train context-adaptive SASRec/LLM fusion model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--title_batch_size", type=int, default=128)
    parser.add_argument("--precompute_batch_size", type=int, default=24)
    parser.add_argument("--train_batch_size", type=int, default=192)
    parser.add_argument("--eval_batch_size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--residual_scale", type=float, default=0.20)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--cache_tag", type=str, default="full")
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_val", type=int, default=0)
    parser.add_argument("--limit_test", type=int, default=0)
    return parser.parse_args()


@dataclass
class SplitData:
    prompts: List[str]
    histories: List[List[int]]
    targets: List[int]
    user_ids: List[str]
    target_item_ids: List[str]


def load_item_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(ITEM_MAPPING_PATH, "r") as f:
        mapping_data = json.load(f)
    item_to_id = mapping_data["item_to_id"]
    id_to_item = {v: k for k, v in item_to_id.items()}
    return item_to_id, id_to_item


def load_item_titles(item_to_id: Dict[str, int]) -> Tuple[List[str], List[str]]:
    with open(TITLES_PATH, "r", encoding="utf-8") as f:
        titles_data = json.load(f)

    title_lookup = {item["item_id"]: item["condensed_title"] for item in titles_data}
    ordered_item_ids = [None] * len(item_to_id)
    ordered_titles = [None] * len(item_to_id)
    for item_id, internal_id in item_to_id.items():
        ordered_item_ids[internal_id] = item_id
        ordered_titles[internal_id] = title_lookup.get(item_id, "Unknown")
    return ordered_item_ids, ordered_titles


def load_aligned_split(
    llm_split_name: str,
    sasrec_filename: str,
    id_to_item: Dict[int, str],
    limit: int = 0,
) -> SplitData:
    llm_path = os.path.join(LLM_DATA_DIR, f"{llm_split_name}.json")
    sasrec_path = os.path.join(SASREC_SPLIT_DIR, sasrec_filename)

    with open(llm_path, "r") as f:
        llm_data = json.load(f)
    with open(sasrec_path, "r") as f:
        sasrec_data = json.load(f)

    llm_lookup = {(x["user_id"], x["target_item_id"]): x for x in llm_data}

    prompts = []
    histories = []
    targets = []
    user_ids = []
    target_item_ids = []

    for sample in sasrec_data:
        target_internal = int(sample["ground_truth"])
        target_item_id = id_to_item.get(target_internal)
        if target_item_id is None:
            continue

        key = (sample["user_id"], target_item_id)
        llm_sample = llm_lookup.get(key)
        if llm_sample is None:
            continue

        prompts.append(llm_sample["prompt"])
        histories.append([int(x) for x in sample["prompt"].split()])
        targets.append(target_internal)
        user_ids.append(sample["user_id"])
        target_item_ids.append(target_item_id)

        if limit and len(prompts) >= limit:
            break

    return SplitData(
        prompts=prompts,
        histories=histories,
        targets=targets,
        user_ids=user_ids,
        target_item_ids=target_item_ids,
    )


def load_all_splits(id_to_item: Dict[int, str], args: argparse.Namespace) -> Dict[str, SplitData]:
    splits = {
        "train": load_aligned_split("train", "train.json", id_to_item, args.limit_train),
        "val": load_aligned_split("val", "validation.json", id_to_item, args.limit_val),
        "test": load_aligned_split("test", "test.json", id_to_item, args.limit_test),
    }
    print("Aligned samples:")
    for split_name, split in splits.items():
        print(f"  {split_name:<5}: {len(split.prompts)}")
    return splits


def load_sasrec_model(device: torch.device) -> Tuple[SasRecForCausalLM, PreTrainedTokenizerFast]:
    model = SasRecForCausalLM.from_pretrained(SASREC_CHECKPOINT).to(device).eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(SASREC_CHECKPOINT)
    tokenizer.padding_side = "left"
    return model, tokenizer


def load_llm_model(device: torch.device) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    return model, tokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def gather_last_token(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_pos = attention_mask.long().sum(dim=1) - 1
    batch_idx = torch.arange(hidden_state.size(0), device=hidden_state.device)
    return hidden_state[batch_idx, last_pos]


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    x_min = x.min(dim=1, keepdim=True).values
    x_max = x.max(dim=1, keepdim=True).values
    return (x - x_min) / (x_max - x_min).clamp_min(1e-6)


def clean_prediction_text(text: str) -> str:
    text = text.strip()
    if "Next product:" in text:
        text = text.split("Next product:", 1)[-1].strip()
    for marker in ["\n", "- Review:", "(Rating:", "Rating:"]:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    text = text.strip(" \"'.,:;-")
    words = text.split()
    if len(words) > 12:
        text = " ".join(words[:12])
    return text if text else "Unknown"


def compute_ndcg_at_k(topk: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    matches = topk[:, :k] == targets.unsqueeze(1)
    hit_indices = matches.float().argmax(dim=1)
    hits = matches.any(dim=1)
    denom = torch.log2(hit_indices.float() + 2.0)
    ndcg = torch.where(hits, 1.0 / denom, torch.zeros_like(denom))
    return ndcg.mean().item()


def compute_metrics_from_scores(scores: torch.Tensor, targets: torch.Tensor, ks: List[int]) -> Dict[str, float]:
    topk = scores.topk(max(ks), dim=1).indices
    metrics = {}
    for k in ks:
        hits = (topk[:, :k] == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f"HR@{k}"] = hits
    metrics["NDCG@10"] = compute_ndcg_at_k(topk, targets, 10)
    return metrics


def encode_item_titles(
    llm_model: AutoModelForCausalLM,
    llm_tokenizer: AutoTokenizer,
    ordered_titles: List[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    all_embeddings = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(ordered_titles), batch_size), desc="Encoding item titles"):
            batch_titles = ordered_titles[start:start + batch_size]
            inputs = llm_tokenizer(
                batch_titles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=TITLE_MAX_LENGTH,
                pad_to_multiple_of=8,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = llm_model.model(**inputs, use_cache=False)
                pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                pooled = F.normalize(pooled.float(), dim=1)
            all_embeddings.append(pooled.cpu().to(torch.float16))
    return torch.cat(all_embeddings, dim=0)


def compute_sample_stats(
    sas_scores: torch.Tensor,
    llm_scores: torch.Tensor,
    history_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    generated_lengths: torch.Tensor,
) -> torch.Tensor:
    sas_top2 = sas_scores.topk(2, dim=1).values.float()
    llm_top2 = llm_scores.topk(2, dim=1).values.float()
    sas_entropy = -(sas_scores.float().clamp_min(1e-8) * sas_scores.float().clamp_min(1e-8).log()).sum(dim=1)
    llm_std = llm_scores.float().std(dim=1)
    agreement = (sas_scores.argmax(dim=1) == llm_scores.argmax(dim=1)).float()
    stats = torch.stack(
        [
            history_lengths.float(),
            prompt_lengths.float(),
            generated_lengths.float(),
            sas_entropy,
            sas_top2[:, 0] - sas_top2[:, 1],
            llm_top2[:, 0] - llm_top2[:, 1],
            llm_std,
            agreement,
        ],
        dim=1,
    )
    return stats

def get_cache_dir(args: argparse.Namespace) -> str:
    return os.path.join(SCRIPT_DIR, "cache", f"context_adaptive_v1_{args.cache_tag}")


def cache_path_for_split(split_name: str, args: argparse.Namespace) -> str:
    return os.path.join(get_cache_dir(args), f"{split_name}_features.pt")


def precompute_split_features(
    split_name: str,
    split_data: SplitData,
    sasrec_model: SasRecForCausalLM,
    sasrec_tokenizer: PreTrainedTokenizerFast,
    llm_model: AutoModelForCausalLM,
    llm_tokenizer: AutoTokenizer,
    item_embeddings: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    os.makedirs(get_cache_dir(args), exist_ok=True)
    cache_path = cache_path_for_split(split_name, args)
    if os.path.exists(cache_path) and not args.force_recompute:
        print(f"Loading cached features: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    item_embeddings = item_embeddings.to(device)
    all_sas_scores = []
    all_llm_scores = []
    all_context_hidden = []
    all_stats = []
    all_targets = []
    all_generated_texts = []

    batch_size = args.precompute_batch_size
    with torch.inference_mode():
        for start in tqdm(range(0, len(split_data.prompts), batch_size), desc=f"Precompute {split_name}"):
            end = min(start + batch_size, len(split_data.prompts))
            batch_prompts = split_data.prompts[start:end]
            batch_histories = split_data.histories[start:end]
            batch_targets = split_data.targets[start:end]

            seq_strs = [" ".join(str(x) for x in seq[-100:]) for seq in batch_histories]
            sas_inputs = sasrec_tokenizer(seq_strs, return_tensors="pt", padding=True)
            sas_inputs = {k: v.to(device) for k, v in sas_inputs.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sas_logits = sasrec_model(**sas_inputs).logits[:, -1, 4:]
            sas_scores = F.softmax(sas_logits.float(), dim=-1).to(torch.float16)

            prompt_inputs = llm_tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
                pad_to_multiple_of=8,
            )
            prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prompt_outputs = llm_model(**prompt_inputs, output_hidden_states=True, use_cache=False)
                shallow_hidden = gather_last_token(
                    prompt_outputs.hidden_states[SHALLOW_LAYER_IDX],
                    prompt_inputs["attention_mask"],
                )
                deep_hidden = gather_last_token(
                    prompt_outputs.hidden_states[-1],
                    prompt_inputs["attention_mask"],
                )
                context_hidden = torch.cat([shallow_hidden, deep_hidden], dim=1).float()

            generated = llm_model.generate(
                **prompt_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                use_cache=True,
                pad_token_id=llm_tokenizer.pad_token_id,
                eos_token_id=llm_tokenizer.eos_token_id,
            )
            input_length = prompt_inputs["input_ids"].shape[1]
            predictions = []
            generated_lengths = []
            for row_idx in range(generated.size(0)):
                gen_tokens = generated[row_idx, input_length:]
                generated_lengths.append(int(gen_tokens.numel()))
                pred_text = llm_tokenizer.decode(gen_tokens, skip_special_tokens=True)
                predictions.append(clean_prediction_text(pred_text))

            pred_inputs = llm_tokenizer(
                predictions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=TITLE_MAX_LENGTH,
                pad_to_multiple_of=8,
            )
            pred_inputs = {k: v.to(device) for k, v in pred_inputs.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_outputs = llm_model.model(**pred_inputs, use_cache=False)
                pred_embeddings = mean_pool(pred_outputs.last_hidden_state, pred_inputs["attention_mask"])
                pred_embeddings = F.normalize(pred_embeddings.float(), dim=1)
                llm_scores = pred_embeddings @ item_embeddings.T
                llm_scores = normalize_rows(llm_scores).to(torch.float16)

            stats = compute_sample_stats(
                sas_scores,
                llm_scores,
                history_lengths=torch.tensor([len(x) for x in batch_histories], device=device),
                prompt_lengths=prompt_inputs["attention_mask"].sum(dim=1),
                generated_lengths=torch.tensor(generated_lengths, device=device),
            )

            all_sas_scores.append(sas_scores.cpu())
            all_llm_scores.append(llm_scores.cpu())
            all_context_hidden.append(context_hidden.cpu().to(torch.float16))
            all_stats.append(stats.cpu())
            all_targets.append(torch.tensor(batch_targets, dtype=torch.long))
            all_generated_texts.extend(predictions)

            del sas_inputs, prompt_inputs, pred_inputs
            del sas_logits, sas_scores, prompt_outputs, context_hidden
            del pred_outputs, pred_embeddings, llm_scores, stats, generated

    split_features = {
        "sas_scores": torch.cat(all_sas_scores, dim=0),
        "llm_scores": torch.cat(all_llm_scores, dim=0),
        "context_hidden": torch.cat(all_context_hidden, dim=0),
        "stats": torch.cat(all_stats, dim=0),
        "targets": torch.cat(all_targets, dim=0),
        "generated_texts": all_generated_texts,
        "user_ids": split_data.user_ids,
        "target_item_ids": split_data.target_item_ids,
    }

    torch.save(split_features, cache_path)
    print(f"Saved cached features: {cache_path}")
    return split_features


class ContextAdaptiveFusion(nn.Module):
    def __init__(
        self,
        context_dim: int,
        stats_dim: int,
        stats_mean: torch.Tensor,
        stats_std: torch.Tensor,
        base_alpha: float,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.hidden_proj = nn.Sequential(
            nn.Linear(context_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 192),
            nn.LayerNorm(192),
            nn.GELU(),
        )
        self.stats_proj = nn.Sequential(
            nn.Linear(stats_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.coeff_head = nn.Sequential(
            nn.Linear(192 + 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 8),
        )
        self.base_alpha = base_alpha
        self.residual_scale = residual_scale
        self.register_buffer("stats_mean", stats_mean)
        self.register_buffer("stats_std", stats_std.clamp_min(1e-6))

    def forward(
        self,
        context_hidden: torch.Tensor,
        stats: torch.Tensor,
        sas_scores: torch.Tensor,
        llm_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_stats = (stats - self.stats_mean) / self.stats_std
        context_vec = self.hidden_proj(context_hidden.float())
        stats_vec = self.stats_proj(norm_stats.float())
        coeffs = torch.tanh(self.coeff_head(torch.cat([context_vec, stats_vec], dim=1)))
        weights = coeffs[:, :7]
        bias = 0.05 * coeffs[:, 7:8]

        base_scores = self.base_alpha * sas_scores + (1.0 - self.base_alpha) * llm_scores
        diff = sas_scores - llm_scores
        feature_stack = torch.stack(
            [
                sas_scores,
                llm_scores,
                diff,
                diff.abs(),
                sas_scores * llm_scores,
                sas_scores.square(),
                llm_scores.square(),
            ],
            dim=-1,
        )
        residual = (feature_stack * weights.unsqueeze(1)).sum(dim=-1) + bias
        fused_scores = base_scores + self.residual_scale * residual
        return fused_scores, coeffs


def evaluate_fixed_alpha(
    sas_scores: torch.Tensor,
    llm_scores: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    batch_size: int,
) -> Dict[str, float]:
    weighted_sums = None
    total_rows = 0
    for start in range(0, targets.size(0), batch_size):
        end = min(start + batch_size, targets.size(0))
        scores = alpha * sas_scores[start:end] + (1.0 - alpha) * llm_scores[start:end]
        batch_metrics = compute_metrics_from_scores(scores, targets[start:end], [1, 5, 10, 20])
        batch_rows = end - start
        if weighted_sums is None:
            weighted_sums = {k: 0.0 for k in batch_metrics}
        for key, value in batch_metrics.items():
            weighted_sums[key] += value * batch_rows
        total_rows += batch_rows
    return {key: value / total_rows for key, value in weighted_sums.items()}


def find_best_fixed_alpha(
    val_sas: torch.Tensor,
    val_llm: torch.Tensor,
    val_targets: torch.Tensor,
    batch_size: int,
) -> Tuple[float, Dict[str, float]]:
    best_alpha = 0.7
    best_metrics = None
    for alpha in [i / 20.0 for i in range(0, 21)]:
        metrics = evaluate_fixed_alpha(val_sas, val_llm, val_targets, alpha, batch_size)
        if best_metrics is None:
            best_alpha = alpha
            best_metrics = metrics
            continue
        better = metrics["HR@1"] > best_metrics["HR@1"]
        same_hr1 = math.isclose(metrics["HR@1"], best_metrics["HR@1"], rel_tol=0.0, abs_tol=1e-8)
        if better or (same_hr1 and metrics["HR@10"] > best_metrics["HR@10"]):
            best_alpha = alpha
            best_metrics = metrics
    return best_alpha, best_metrics


def move_split_to_device(split_features: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    tensor_keys = ["sas_scores", "llm_scores", "context_hidden", "stats", "targets"]
    moved = {}
    for key, value in split_features.items():
        if key in tensor_keys:
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def evaluate_model(
    model: ContextAdaptiveFusion,
    split: Dict[str, torch.Tensor],
    batch_size: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    metric_sums = None
    coeff_sum = None
    total_rows = 0
    with torch.inference_mode():
        for start in range(0, split["targets"].size(0), batch_size):
            end = min(start + batch_size, split["targets"].size(0))
            scores, coeffs = model(
                split["context_hidden"][start:end],
                split["stats"][start:end],
                split["sas_scores"][start:end],
                split["llm_scores"][start:end],
            )
            batch_metrics = compute_metrics_from_scores(scores.float(), split["targets"][start:end], [1, 5, 10, 20])
            batch_rows = end - start
            if metric_sums is None:
                metric_sums = {k: 0.0 for k in batch_metrics}
            for key, value in batch_metrics.items():
                metric_sums[key] += value * batch_rows
            coeff_batch = coeffs.float().sum(dim=0)
            coeff_sum = coeff_batch if coeff_sum is None else coeff_sum + coeff_batch
            total_rows += batch_rows

    metrics = {key: value / total_rows for key, value in metric_sums.items()}
    coeff_mean = (coeff_sum / total_rows).tolist()
    coeff_stats = {f"coeff_{idx}": coeff_mean[idx] for idx in range(len(coeff_mean))}
    return metrics, coeff_stats


def train_model(
    model: ContextAdaptiveFusion,
    train_split: Dict[str, torch.Tensor],
    val_split: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[ContextAdaptiveFusion, List[Dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    history = []
    best_state = None
    best_val_hr1 = -1.0
    best_val_hr10 = -1.0
    patience_counter = 0
    num_train = train_split["targets"].size(0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(num_train, device=train_split["targets"].device)
        train_loss_sum = 0.0
        train_hr1_sum = 0.0
        num_batches = 0

        for start in range(0, num_train, args.train_batch_size):
            end = min(start + args.train_batch_size, num_train)
            idx = perm[start:end]
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores, coeffs = model(
                    train_split["context_hidden"][idx],
                    train_split["stats"][idx],
                    train_split["sas_scores"][idx],
                    train_split["llm_scores"][idx],
                )
                ce_loss = F.cross_entropy(
                    scores.float(),
                    train_split["targets"][idx],
                    label_smoothing=args.label_smoothing,
                )
                reg_loss = coeffs.square().mean()
                loss = ce_loss + 0.002 * reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.inference_mode():
                top1 = scores.argmax(dim=1)
                hr1 = (top1 == train_split["targets"][idx]).float().mean().item()

            train_loss_sum += loss.item()
            train_hr1_sum += hr1
            num_batches += 1

        train_loss = train_loss_sum / max(num_batches, 1)
        train_hr1 = train_hr1_sum / max(num_batches, 1)
        val_metrics, coeff_stats = evaluate_model(model, val_split, args.eval_batch_size)
        scheduler.step(val_metrics["HR@1"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_hr1": train_hr1,
            "val_hr1": val_metrics["HR@1"],
            "val_hr10": val_metrics["HR@10"],
            "val_ndcg10": val_metrics["NDCG@10"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        epoch_record.update(coeff_stats)
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} train_hr1={train_hr1:.4f} "
            f"val_hr1={val_metrics['HR@1']:.4f} val_hr10={val_metrics['HR@10']:.4f}"
        )

        improved = val_metrics["HR@1"] > best_val_hr1
        same_hr1 = math.isclose(val_metrics["HR@1"], best_val_hr1, rel_tol=0.0, abs_tol=1e-8)
        if improved or (same_hr1 and val_metrics["HR@10"] > best_val_hr10):
            best_val_hr1 = val_metrics["HR@1"]
            best_val_hr10 = val_metrics["HR@10"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(train_split["targets"].device) for k, v in best_state.items()})
    return model, history


def main() -> None:
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(get_cache_dir(args), exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_global_seed(args.seed)

    device = torch.device(args.device)
    print("=" * 72)
    print("Context-Adaptive Fusion Training")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    item_to_id, id_to_item = load_item_mapping()
    ordered_item_ids, ordered_titles = load_item_titles(item_to_id)
    splits = load_all_splits(id_to_item, args)

    print("\nLoading models...")
    sasrec_model, sasrec_tokenizer = load_sasrec_model(device)
    llm_model, llm_tokenizer = load_llm_model(device)

    print("\nEncoding catalog titles...")
    item_embeddings = encode_item_titles(
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        ordered_titles=ordered_titles,
        batch_size=args.title_batch_size,
        device=device,
    )
    torch.save(
        {
            "item_ids": ordered_item_ids,
            "item_embeddings": item_embeddings,
        },
        os.path.join(get_cache_dir(args), "item_embeddings.pt"),
    )

    print("\nPrecomputing split features...")
    split_features_cpu = {}
    for split_name, split_data in splits.items():
        split_features_cpu[split_name] = precompute_split_features(
            split_name=split_name,
            split_data=split_data,
            sasrec_model=sasrec_model,
            sasrec_tokenizer=sasrec_tokenizer,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            item_embeddings=item_embeddings,
            args=args,
            device=device,
        )

    del sasrec_model, llm_model, item_embeddings
    gc.collect()
    torch.cuda.empty_cache()

    print("\nMoving cached tensors to GPU...")
    split_features = {
        split_name: move_split_to_device(features, device)
        for split_name, features in split_features_cpu.items()
    }

    print("\nSearching best fixed alpha on validation set...")
    base_alpha, val_fixed_metrics = find_best_fixed_alpha(
        split_features["val"]["sas_scores"],
        split_features["val"]["llm_scores"],
        split_features["val"]["targets"],
        args.eval_batch_size,
    )
    test_fixed_metrics = evaluate_fixed_alpha(
        split_features["test"]["sas_scores"],
        split_features["test"]["llm_scores"],
        split_features["test"]["targets"],
        base_alpha,
        args.eval_batch_size,
    )
    sasrec_test_metrics = evaluate_fixed_alpha(
        split_features["test"]["sas_scores"],
        split_features["test"]["llm_scores"],
        split_features["test"]["targets"],
        1.0,
        args.eval_batch_size,
    )
    llm_test_metrics = evaluate_fixed_alpha(
        split_features["test"]["sas_scores"],
        split_features["test"]["llm_scores"],
        split_features["test"]["targets"],
        0.0,
        args.eval_batch_size,
    )

    print(f"Best fixed alpha on val: {base_alpha:.2f}")
    print(f"  Fixed fusion test HR@1={test_fixed_metrics['HR@1']:.4f} HR@10={test_fixed_metrics['HR@10']:.4f}")

    model = ContextAdaptiveFusion(
        context_dim=split_features["train"]["context_hidden"].size(1),
        stats_dim=split_features["train"]["stats"].size(1),
        stats_mean=split_features["train"]["stats"].float().mean(dim=0),
        stats_std=split_features["train"]["stats"].float().std(dim=0),
        base_alpha=base_alpha,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
    ).to(device)

    print("\nTraining dynamic fusion model...")
    model, history = train_model(
        model=model,
        train_split=split_features["train"],
        val_split=split_features["val"],
        args=args,
    )

    print("\nEvaluating best checkpoint...")
    val_metrics, val_coeff_stats = evaluate_model(model, split_features["val"], args.eval_batch_size)
    test_metrics, test_coeff_stats = evaluate_model(model, split_features["test"], args.eval_batch_size)

    model_path = os.path.join(RESULTS_DIR, "context_adaptive_fusion_model.pt")
    torch.save(model.state_dict(), model_path)

    results = {
        "config": vars(args),
        "num_items": len(ordered_item_ids),
        "split_sizes": {k: len(v.prompts) for k, v in splits.items()},
        "baseline": {
            "sasrec_test": sasrec_test_metrics,
            "llm_test": llm_test_metrics,
            "best_fixed_alpha": base_alpha,
            "fixed_fusion_val": val_fixed_metrics,
            "fixed_fusion_test": test_fixed_metrics,
        },
        "dynamic_fusion": {
            "val": val_metrics,
            "test": test_metrics,
            "val_coeff_mean": val_coeff_stats,
            "test_coeff_mean": test_coeff_stats,
            "history": history,
            "model_path": model_path,
        },
    }

    results_path = os.path.join(RESULTS_DIR, "context_adaptive_fusion_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 72)
    print("Final Results")
    print("=" * 72)
    print("Test baselines:")
    print(
        f"  SASRec      HR@1={sasrec_test_metrics['HR@1']:.4f} "
        f"HR@10={sasrec_test_metrics['HR@10']:.4f}"
    )
    print(
        f"  LLM         HR@1={llm_test_metrics['HR@1']:.4f} "
        f"HR@10={llm_test_metrics['HR@10']:.4f}"
    )
    print(
        f"  Fixed alpha HR@1={test_fixed_metrics['HR@1']:.4f} "
        f"HR@10={test_fixed_metrics['HR@10']:.4f} alpha={base_alpha:.2f}"
    )
    print("Dynamic fusion:")
    print(
        f"  HR@1={test_metrics['HR@1']:.4f} "
        f"HR@5={test_metrics['HR@5']:.4f} "
        f"HR@10={test_metrics['HR@10']:.4f} "
        f"NDCG@10={test_metrics['NDCG@10']:.4f}"
    )
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
