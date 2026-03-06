#!/usr/bin/env python3

import gc
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(ROOT_DIR)

sys.path.append(os.path.join(REPO_ROOT, "Rec-Transformer"))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(ROOT_DIR, "4_finetune_to_upgrade_LLM"))

from sasrec import SasRecForCausalLM  # type: ignore
from pipeline_utils import format_chat_prompt, normalize_title  # type: ignore


ITEM_MAPPING_PATH = os.path.join(ROOT_DIR, "vanilla_sasrec", "processed_data", "item_mapping.json")
SASREC_SPLIT_DIR = os.path.join(ROOT_DIR, "vanilla_sasrec", "processed_data", "splits")
SASREC_CHECKPOINT = os.path.join(
    ROOT_DIR,
    "vanilla_sasrec",
    "checkpoints",
    "sasrec_beauty_20260226_055626",
    "best_model",
)
SFT_DATA_DIR = os.path.join(ROOT_DIR, "4_finetune_to_upgrade_LLM", "data")
SFT_CHECKPOINT = os.path.join(ROOT_DIR, "4_finetune_to_upgrade_LLM", "outputs", "qwen3_title_sft", "best_model")
TITLES_PATH = os.path.join(ROOT_DIR, "use_Qwen3-1-7B_to_generate_title", "item_titles_unique.json")

CACHE_ROOT = os.path.join(SCRIPT_DIR, "cache")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
JOBS_DIR = os.path.join(SCRIPT_DIR, "jobs")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")

TITLE_MAX_LENGTH = 64
SHALLOW_LAYER_IDX = 8


@dataclass
class SplitData:
    prompts: List[str]
    histories: List[List[int]]
    targets: List[int]
    user_ids: List[str]
    target_item_ids: List[str]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cache_dir(cache_tag: str) -> str:
    return os.path.join(CACHE_ROOT, cache_tag)


def cache_path(cache_tag: str, name: str) -> str:
    return os.path.join(get_cache_dir(cache_tag), name)


def success_path(cache_tag: str) -> str:
    return cache_path(cache_tag, "_SUCCESS.json")


def wait_for_cache_ready(cache_tag: str, poll_seconds: int = 30) -> dict:
    path = success_path(cache_tag)
    while not os.path.exists(path):
        time.sleep(poll_seconds)
    return load_json(path)


def load_item_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    mapping_data = load_json(ITEM_MAPPING_PATH)
    item_to_id = mapping_data["item_to_id"]
    id_to_item = {v: k for k, v in item_to_id.items()}
    return item_to_id, id_to_item


def load_item_titles(item_to_id: Dict[str, int]) -> Tuple[List[str], List[str]]:
    titles_data = load_json(TITLES_PATH)
    title_lookup = {item["item_id"]: item["condensed_title"] for item in titles_data}
    ordered_item_ids = [None] * len(item_to_id)
    ordered_titles = [None] * len(item_to_id)
    for item_id, internal_id in item_to_id.items():
        ordered_item_ids[internal_id] = item_id
        ordered_titles[internal_id] = title_lookup.get(item_id, "Unknown")
    return ordered_item_ids, ordered_titles


def load_aligned_split(
    sft_split_name: str,
    sasrec_filename: str,
    item_to_id: Dict[str, int],
    limit: int = 0,
) -> SplitData:
    sft_rows = load_jsonl(os.path.join(SFT_DATA_DIR, f"{sft_split_name}.jsonl"))
    sasrec_rows = load_json(os.path.join(SASREC_SPLIT_DIR, sasrec_filename))
    sasrec_lookup = {(row["user_id"], row["ground_truth"]): row for row in sasrec_rows}

    prompts = []
    histories = []
    targets = []
    user_ids = []
    target_item_ids = []

    for row in sft_rows:
        target_item_id = row["target_item_id"]
        internal_id = item_to_id.get(target_item_id)
        if internal_id is None:
            continue
        sasrec_row = sasrec_lookup.get((row["user_id"], str(internal_id)))
        if sasrec_row is None:
            continue

        prompts.append(row["prompt"])
        histories.append([int(x) for x in sasrec_row["prompt"].split()])
        targets.append(internal_id)
        user_ids.append(row["user_id"])
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


def load_all_splits(item_to_id: Dict[str, int], limit_train: int = 0, limit_val: int = 0, limit_test: int = 0) -> Dict[str, SplitData]:
    return {
        "train": load_aligned_split("train", "train.json", item_to_id, limit_train),
        "val": load_aligned_split("val", "validation.json", item_to_id, limit_val),
        "test": load_aligned_split("test", "test.json", item_to_id, limit_test),
    }


def load_sasrec_model(device: torch.device) -> Tuple[SasRecForCausalLM, PreTrainedTokenizerFast]:
    model = SasRecForCausalLM.from_pretrained(SASREC_CHECKPOINT).to(device).eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(SASREC_CHECKPOINT)
    tokenizer.padding_side = "left"
    return model, tokenizer


def load_sft_model(device: torch.device) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": {"": device.index if device.index is not None else 0},
    }
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        model_kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(SFT_CHECKPOINT, **model_kwargs).eval()
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
    text = normalize_title(text)
    return text if text else "Unknown"


def build_chat_prompts(prompts: List[str]) -> List[str]:
    return [format_chat_prompt(prompt) for prompt in prompts]


@torch.inference_mode()
def encode_item_titles(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ordered_titles: List[str],
    batch_size: int,
    device: torch.device,
    max_length: int = TITLE_MAX_LENGTH,
) -> torch.Tensor:
    backbone = model.model if hasattr(model, "model") else model
    all_embeddings = []
    for start in range(0, len(ordered_titles), batch_size):
        batch_titles = ordered_titles[start:start + batch_size]
        inputs = tokenizer(
            batch_titles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            pad_to_multiple_of=8,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = backbone(**inputs, use_cache=False)
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
    return torch.stack(
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


@torch.inference_mode()
def precompute_split_features(
    split_name: str,
    split_data: SplitData,
    sasrec_model: SasRecForCausalLM,
    sasrec_tokenizer: PreTrainedTokenizerFast,
    sft_model: AutoModelForCausalLM,
    sft_tokenizer: AutoTokenizer,
    item_embeddings: torch.Tensor,
    device: torch.device,
    batch_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
) -> Dict[str, torch.Tensor]:
    item_embeddings = item_embeddings.to(device)
    all_sas_scores = []
    all_llm_scores = []
    all_context_hidden = []
    all_stats = []
    all_targets = []
    all_generated_texts = []

    for start in range(0, len(split_data.prompts), batch_size):
        end = min(start + batch_size, len(split_data.prompts))
        batch_prompts = build_chat_prompts(split_data.prompts[start:end])
        batch_histories = split_data.histories[start:end]
        batch_targets = split_data.targets[start:end]

        seq_strs = [" ".join(str(x) for x in seq[-100:]) for seq in batch_histories]
        sas_inputs = sasrec_tokenizer(seq_strs, return_tensors="pt", padding=True)
        sas_inputs = {k: v.to(device) for k, v in sas_inputs.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sas_logits = sasrec_model(**sas_inputs).logits[:, -1, 4:]
        sas_scores = F.softmax(sas_logits.float(), dim=-1).to(torch.float16)

        prompt_inputs = sft_tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            pad_to_multiple_of=8,
        )
        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prompt_outputs = sft_model(**prompt_inputs, output_hidden_states=True, use_cache=False)
            shallow_hidden = gather_last_token(prompt_outputs.hidden_states[SHALLOW_LAYER_IDX], prompt_inputs["attention_mask"])
            deep_hidden = gather_last_token(prompt_outputs.hidden_states[-1], prompt_inputs["attention_mask"])
            context_hidden = torch.cat([shallow_hidden, deep_hidden], dim=1).float()

        generated = sft_model.generate(
            **prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=sft_tokenizer.pad_token_id,
            eos_token_id=sft_tokenizer.eos_token_id,
        )
        input_length = prompt_inputs["input_ids"].shape[1]
        predictions = []
        generated_lengths = []
        for row_idx in range(generated.size(0)):
            gen_tokens = generated[row_idx, input_length:]
            generated_lengths.append(int(gen_tokens.numel()))
            pred_text = sft_tokenizer.decode(gen_tokens, skip_special_tokens=True)
            predictions.append(clean_prediction_text(pred_text))

        pred_inputs = sft_tokenizer(
            predictions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=TITLE_MAX_LENGTH,
            pad_to_multiple_of=8,
        )
        pred_inputs = {k: v.to(device) for k, v in pred_inputs.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            backbone = sft_model.model if hasattr(sft_model, "model") else sft_model
            pred_outputs = backbone(**pred_inputs, use_cache=False)
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
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "sas_scores": torch.cat(all_sas_scores, dim=0),
        "llm_scores": torch.cat(all_llm_scores, dim=0),
        "context_hidden": torch.cat(all_context_hidden, dim=0),
        "stats": torch.cat(all_stats, dim=0),
        "targets": torch.cat(all_targets, dim=0),
        "generated_texts": all_generated_texts,
        "user_ids": split_data.user_ids,
        "target_item_ids": split_data.target_item_ids,
    }


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


def move_split_to_device(split_features: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    tensor_keys = ["sas_scores", "llm_scores", "context_hidden", "stats", "targets"]
    moved = {}
    for key, value in split_features.items():
        if key in tensor_keys:
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


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
    coarse = [i / 20.0 for i in range(0, 21)]
    best_alpha = 0.5
    best_metrics = None
    for alpha in coarse:
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

    fine_start = max(0.0, best_alpha - 0.10)
    fine_end = min(1.0, best_alpha + 0.10)
    fine = [round(fine_start + 0.01 * i, 2) for i in range(int(round((fine_end - fine_start) / 0.01)) + 1)]
    for alpha in fine:
        metrics = evaluate_fixed_alpha(val_sas, val_llm, val_targets, alpha, batch_size)
        better = metrics["HR@1"] > best_metrics["HR@1"]
        same_hr1 = math.isclose(metrics["HR@1"], best_metrics["HR@1"], rel_tol=0.0, abs_tol=1e-8)
        if better or (same_hr1 and metrics["HR@10"] > best_metrics["HR@10"]):
            best_alpha = alpha
            best_metrics = metrics

    return best_alpha, best_metrics
