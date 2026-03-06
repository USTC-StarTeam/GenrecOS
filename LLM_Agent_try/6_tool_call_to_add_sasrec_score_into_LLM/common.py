#!/usr/bin/env python3

import gc
import json
import math
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(ROOT_DIR)

sys.path.append(os.path.join(REPO_ROOT, "Rec-Transformer"))

from sasrec import SasRecForCausalLM  # type: ignore


RAW_DATA_PATH = os.path.join(REPO_ROOT, "Data", "Amazons", "data", "All_Beauty.jsonl")
BASE_MODEL_PATH = os.path.join(REPO_ROOT, "LLM4RecPart", "models", "Qwen3-1-7B")
SFT_BEST_MODEL_PATH = os.path.join(ROOT_DIR, "4_finetune_to_upgrade_LLM", "outputs", "qwen3_title_sft", "best_model")
SFT_DATA_DIR = os.path.join(ROOT_DIR, "4_finetune_to_upgrade_LLM", "data")
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

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
JOBS_DIR = os.path.join(SCRIPT_DIR, "jobs")
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")

TOOL_TOKEN = "[tool:seqscore]"
TITLE_MAX_LENGTH = 64
DEFAULT_MAX_PROMPT_LENGTH = 1232

ORIGINAL_SYSTEM_PROMPT = (
    "You are a recommendation model. Given a user's purchase history, predict the "
    "single most likely next product title. Respond with only the next product "
    "title and nothing else."
)

TOOL_SYSTEM_PROMPT = (
    "You are a recommendation model. Given a user's purchase history, predict the "
    "single most likely next product title. If you need sequence-score help, first "
    f"output {TOOL_TOKEN} and then the title. Otherwise output only the title."
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: List[dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_title(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    return text


def clean_prediction_text(text: str) -> str:
    text = text.strip()
    if "Next product:" in text:
        text = text.split("Next product:", 1)[-1].strip()
    for marker in ["\n", "- Review:", "(Rating:", "Rating:", IM_END]:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    text = text.strip(" \"'.,:;-")
    words = text.split()
    if len(words) > 12:
        text = " ".join(words[:12])
    return normalize_title(text)


def strip_tool_token(text: str) -> Tuple[bool, str]:
    text = text.strip()
    if text.startswith(TOOL_TOKEN):
        stripped = text[len(TOOL_TOKEN):].strip()
        return True, clean_prediction_text(stripped) or "Unknown"
    return False, clean_prediction_text(text) or "Unknown"


def has_tool_token(text: str) -> bool:
    return text.strip().startswith(TOOL_TOKEN)


def original_chat_prompt(prompt: str) -> str:
    return (
        f"{IM_START}system\n{ORIGINAL_SYSTEM_PROMPT}{IM_END}\n"
        f"{IM_START}user\n{prompt}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def tool_chat_prompt(prompt: str) -> str:
    return (
        f"{IM_START}system\n{TOOL_SYSTEM_PROMPT}{IM_END}\n"
        f"{IM_START}user\n{prompt}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def tool_chat_example(prompt: str, assistant_target: str) -> str:
    return f"{tool_chat_prompt(prompt)}{assistant_target}{IM_END}\n"


def choose_model_path(track_name: str) -> str:
    if track_name == "pre_sft":
        return BASE_MODEL_PATH
    if track_name == "post_sft":
        return SFT_BEST_MODEL_PATH
    raise ValueError(f"Unsupported track_name: {track_name}")


def get_track_dir(track_name: str) -> str:
    return os.path.join(DATA_DIR, track_name)


def get_track_output_dir(track_name: str) -> str:
    return os.path.join(OUTPUTS_DIR, track_name)


def get_track_best_model_dir(track_name: str) -> str:
    return os.path.join(get_track_output_dir(track_name), "best_model")


def teacher_summary_path(track_name: str) -> str:
    return os.path.join(get_track_dir(track_name), "teacher_summary.json")


def evaluation_result_path(track_name: str) -> str:
    return os.path.join(RESULTS_DIR, f"{track_name}_tool_eval.json")


def wait_for_path(path: str, poll_seconds: int = 30) -> None:
    while not os.path.exists(path):
        time.sleep(poll_seconds)


def load_item_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    mapping_data = load_json(ITEM_MAPPING_PATH)
    item_to_id = mapping_data["item_to_id"]
    id_to_item = {int(v): k for k, v in item_to_id.items()}
    return item_to_id, id_to_item


def load_ordered_titles(item_to_id: Dict[str, int]) -> Tuple[List[str], List[str]]:
    titles_data = load_json(TITLES_PATH)
    title_lookup = {row["item_id"]: normalize_title(row["condensed_title"]) for row in titles_data}
    ordered_item_ids = [None] * len(item_to_id)
    ordered_titles = [None] * len(item_to_id)
    for item_id, internal_id in item_to_id.items():
        ordered_item_ids[internal_id] = item_id
        ordered_titles[internal_id] = title_lookup[item_id]
    return ordered_item_ids, ordered_titles


def load_aligned_rows(split_name: str, item_to_id: Dict[str, int], limit: int = 0) -> List[dict]:
    sft_rows = load_jsonl(os.path.join(SFT_DATA_DIR, f"{split_name}.jsonl"))
    sasrec_file = {"train": "train.json", "val": "validation.json", "test": "test.json"}[split_name]
    sasrec_rows = load_json(os.path.join(SASREC_SPLIT_DIR, sasrec_file))
    sasrec_lookup = {(row["user_id"], row["ground_truth"]): row for row in sasrec_rows}

    aligned_rows: List[dict] = []
    for row in sft_rows:
        target_item_id = row["target_item_id"]
        internal_id = item_to_id.get(target_item_id)
        if internal_id is None:
            continue
        sasrec_row = sasrec_lookup.get((row["user_id"], str(internal_id)))
        if sasrec_row is None:
            continue
        merged = dict(row)
        merged["target_internal_id"] = internal_id
        merged["sasrec_history_ids"] = [int(x) for x in sasrec_row["prompt"].split()] if sasrec_row["prompt"] else []
        aligned_rows.append(merged)
        if limit and len(aligned_rows) >= limit:
            break
    return aligned_rows


def load_sasrec_model(device: torch.device) -> Tuple[SasRecForCausalLM, PreTrainedTokenizerFast]:
    model = SasRecForCausalLM.from_pretrained(SASREC_CHECKPOINT).to(device).eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(SASREC_CHECKPOINT)
    tokenizer.padding_side = "left"
    return model, tokenizer


def load_causal_model(
    model_path: str,
    device: torch.device,
    add_tool_token: bool,
    train_mode: bool = False,
    gradient_checkpointing: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "right" if train_mode else "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if add_tool_token and TOOL_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_tokens([TOOL_TOKEN])

    model_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if not train_mode:
        model_kwargs["device_map"] = {"": device.index if device.index is not None else 0}
    if torch.cuda.is_available() and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if add_tool_token:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = not gradient_checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not train_mode:
        model.eval()
    return model, tokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def normalize_rows(scores: torch.Tensor) -> torch.Tensor:
    score_min = scores.min(dim=1, keepdim=True).values
    score_max = scores.max(dim=1, keepdim=True).values
    return (scores - score_min) / (score_max - score_min).clamp_min(1e-6)


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
    outputs: List[torch.Tensor] = []
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
            model_outputs = backbone(**inputs, use_cache=False)
            pooled = mean_pool(model_outputs.last_hidden_state, inputs["attention_mask"])
            pooled = F.normalize(pooled.float(), dim=1)
        outputs.append(pooled.cpu().to(torch.float16))
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def compute_sasrec_scores(
    sasrec_model: SasRecForCausalLM,
    sasrec_tokenizer: PreTrainedTokenizerFast,
    histories: List[List[int]],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    for start in range(0, len(histories), batch_size):
        batch = histories[start:start + batch_size]
        seq_text = [" ".join(str(x) for x in seq[-100:]) for seq in batch]
        inputs = sasrec_tokenizer(seq_text, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = sasrec_model(**inputs).logits[:, -1, 4:]
        outputs.append(F.softmax(logits.float(), dim=-1).cpu().to(torch.float16))
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def generate_predictions_and_scores(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rows: List[dict],
    item_embeddings: torch.Tensor,
    batch_size: int,
    device: torch.device,
    max_prompt_length: int,
    max_new_tokens: int,
    prompt_style: str,
) -> Dict[str, object]:
    assert prompt_style in {"original", "tool"}
    item_embeddings = item_embeddings.to(device)
    backbone = model.model if hasattr(model, "model") else model

    all_scores: List[torch.Tensor] = []
    raw_texts: List[str] = []
    stripped_titles: List[str] = []
    tool_flags: List[bool] = []

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        if prompt_style == "original":
            prompts = [original_chat_prompt(row["prompt"]) for row in batch_rows]
        else:
            prompts = [tool_chat_prompt(row["prompt"]) for row in batch_rows]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            pad_to_multiple_of=8,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        batch_raw_texts = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        batch_flags: List[bool] = []
        batch_titles: List[str] = []
        for text in batch_raw_texts:
            raw = text.strip()
            flag, stripped = strip_tool_token(raw)
            raw_texts.append(raw)
            stripped_titles.append(stripped)
            batch_flags.append(flag)
            batch_titles.append(stripped)
            tool_flags.append(flag)

        pred_inputs = tokenizer(
            batch_titles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=TITLE_MAX_LENGTH,
            pad_to_multiple_of=8,
        )
        pred_inputs = {k: v.to(device) for k, v in pred_inputs.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_outputs = backbone(**pred_inputs, use_cache=False)
            pred_embeddings = mean_pool(pred_outputs.last_hidden_state, pred_inputs["attention_mask"])
            pred_embeddings = F.normalize(pred_embeddings.float(), dim=1)
            scores = pred_embeddings @ item_embeddings.T
            scores = normalize_rows(scores).cpu().to(torch.float16)
        all_scores.append(scores)

        del inputs, generated, pred_inputs, pred_outputs, pred_embeddings, scores
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "llm_scores": torch.cat(all_scores, dim=0),
        "raw_predictions": raw_texts,
        "stripped_titles": stripped_titles,
        "tool_flags": tool_flags,
    }


def compute_label_summary(
    llm_scores: torch.Tensor,
    sas_scores: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    llm_top1 = llm_scores.argmax(dim=1)
    sas_top1 = sas_scores.argmax(dim=1)
    llm_hit1 = llm_top1 == targets
    sas_hit1 = sas_top1 == targets
    tool_labels = (~llm_hit1) & sas_hit1
    return {
        "llm_top1": llm_top1,
        "sas_top1": sas_top1,
        "llm_hit1": llm_hit1,
        "sas_hit1": sas_hit1,
        "tool_labels": tool_labels,
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
        metrics[f"HR@{k}"] = (topk[:, :k] == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    metrics["NDCG@10"] = compute_ndcg_at_k(topk, targets, 10)
    return metrics


def evaluate_fixed_alpha(
    sas_scores: torch.Tensor,
    llm_scores: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> Dict[str, float]:
    scores = alpha * sas_scores.float() + (1.0 - alpha) * llm_scores.float()
    return compute_metrics_from_scores(scores, targets, [1, 5, 10, 20])


def evaluate_routed_alpha(
    sas_scores: torch.Tensor,
    llm_scores: torch.Tensor,
    targets: torch.Tensor,
    route_mask: torch.Tensor,
    alpha: float,
) -> Dict[str, float]:
    fused = alpha * sas_scores.float() + (1.0 - alpha) * llm_scores.float()
    scores = torch.where(route_mask.unsqueeze(1), fused, llm_scores.float())
    return compute_metrics_from_scores(scores, targets, [1, 5, 10, 20])


def find_best_alpha(
    sas_scores: torch.Tensor,
    llm_scores: torch.Tensor,
    targets: torch.Tensor,
    route_mask: Optional[torch.Tensor],
) -> Tuple[float, Dict[str, float]]:
    coarse = [i / 20.0 for i in range(0, 21)]
    best_alpha = 0.0
    best_metrics: Optional[Dict[str, float]] = None
    for alpha in coarse:
        if route_mask is None:
            metrics = evaluate_fixed_alpha(sas_scores, llm_scores, targets, alpha)
        else:
            metrics = evaluate_routed_alpha(sas_scores, llm_scores, targets, route_mask, alpha)
        if best_metrics is None:
            best_alpha = alpha
            best_metrics = metrics
            continue
        same_hr1 = math.isclose(metrics["HR@1"], best_metrics["HR@1"], rel_tol=0.0, abs_tol=1e-8)
        if metrics["HR@1"] > best_metrics["HR@1"] or (same_hr1 and metrics["HR@10"] > best_metrics["HR@10"]):
            best_alpha = alpha
            best_metrics = metrics

    fine_start = max(0.0, best_alpha - 0.10)
    fine_end = min(1.0, best_alpha + 0.10)
    for idx in range(int(round((fine_end - fine_start) / 0.01)) + 1):
        alpha = round(fine_start + idx * 0.01, 2)
        if route_mask is None:
            metrics = evaluate_fixed_alpha(sas_scores, llm_scores, targets, alpha)
        else:
            metrics = evaluate_routed_alpha(sas_scores, llm_scores, targets, route_mask, alpha)
        same_hr1 = math.isclose(metrics["HR@1"], best_metrics["HR@1"], rel_tol=0.0, abs_tol=1e-8)
        if metrics["HR@1"] > best_metrics["HR@1"] or (same_hr1 and metrics["HR@10"] > best_metrics["HR@10"]):
            best_alpha = alpha
            best_metrics = metrics

    return best_alpha, best_metrics


def summarize_tool_behavior(
    generated_flags: List[bool],
    teacher_flags: Optional[List[bool]],
) -> Dict[str, float]:
    generated_tensor = torch.tensor(generated_flags, dtype=torch.bool)
    summary = {
        "generated_tool_rate": generated_tensor.float().mean().item() if generated_tensor.numel() else 0.0,
    }
    if teacher_flags is not None:
        teacher_tensor = torch.tensor(teacher_flags, dtype=torch.bool)
        summary["teacher_tool_rate"] = teacher_tensor.float().mean().item() if teacher_tensor.numel() else 0.0
        summary["route_agreement"] = (generated_tensor == teacher_tensor).float().mean().item() if teacher_tensor.numel() else 0.0
        triggered = generated_tensor.sum().item()
        if triggered:
            summary["route_precision_vs_teacher"] = (generated_tensor & teacher_tensor).float().sum().item() / triggered
        else:
            summary["route_precision_vs_teacher"] = 0.0
        teacher_positive = teacher_tensor.sum().item()
        if teacher_positive:
            summary["route_recall_vs_teacher"] = (generated_tensor & teacher_tensor).float().sum().item() / teacher_positive
        else:
            summary["route_recall_vs_teacher"] = 0.0
    return summary

