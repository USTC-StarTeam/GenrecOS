#!/usr/bin/env python3

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline_utils import format_chat_prompt, load_item_titles, normalize_title


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate finetuned Qwen on title-only next-item prediction.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sft_interest_config.yaml"),
        help="Path to config yaml.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path.",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Optional explicit test jsonl path.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="evaluation",
        help="Prefix for output files in results dir.",
    )
    return parser.parse_args()


def pooled_embeddings(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    show_progress: bool = True,
) -> torch.Tensor:
    all_embeddings = []
    model.eval()
    backbone = model.model if hasattr(model, "model") else model

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Embedding texts")

    for start in iterator:
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = backbone(**inputs, use_cache=False)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        all_embeddings.append(pooled)

    return torch.cat(all_embeddings, dim=0)


def build_prompts(tokenizer, prompts: List[str]) -> Dict[str, torch.Tensor]:
    chat_prompts = [format_chat_prompt(prompt) for prompt in prompts]
    inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True, truncation=True)
    return inputs


def rank_metrics(ranked_indices: torch.Tensor, target_indices: List[int], k_values: List[int]) -> Dict[str, float]:
    results = {}
    total = len(target_indices)

    for k in k_values:
        hits = 0.0
        ndcg = 0.0
        for row_idx, target_idx in enumerate(target_indices):
            row = ranked_indices[row_idx, :k].tolist()
            if target_idx in row:
                hits += 1.0
                rank = row.index(target_idx) + 1
                ndcg += 1.0 / np.log2(rank + 1)
        results[f"HR@{k}"] = hits / total
        results[f"NDCG@{k}"] = ndcg / total

    return results


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths_cfg = config["paths"]
    eval_cfg = config["evaluation"]

    data_dir = os.path.join(script_dir, paths_cfg["data_dir"])
    result_dir = os.path.join(script_dir, paths_cfg["results_dir"])
    os.makedirs(result_dir, exist_ok=True)

    checkpoint_path = args.checkpoint_path or os.path.join(script_dir, eval_cfg["checkpoint_path"])
    titles_path = os.path.join(script_dir, paths_cfg["item_titles_path"])
    test_path = args.test_path or os.path.join(data_dir, "test.jsonl")
    if not os.path.isabs(test_path):
        test_path = os.path.join(script_dir, test_path)

    item_to_title, norm_title_to_items = load_item_titles(titles_path)
    title_texts = list(item_to_title.values())
    item_ids = list(item_to_title.keys())
    item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
        "device_map": {"": 0} if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available() and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
    model.eval()
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, attr):
            setattr(model.generation_config, attr, None)

    item_embeddings = pooled_embeddings(
        model=model,
        tokenizer=tokenizer,
        texts=title_texts,
        batch_size=int(eval_cfg["embedding_batch_size"]),
        max_length=int(eval_cfg["title_max_length"]),
        show_progress=True,
    )
    item_embeddings = F.normalize(item_embeddings, dim=1)

    dataset = load_dataset("json", data_files={"test": test_path})["test"]
    batch_size = int(eval_cfg["batch_size"])
    k_values = [int(k) for k in eval_cfg["k_values"]]
    prediction_rows = []
    all_ranked = []
    all_targets = []
    exact_match_hits = 0

    for start in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[start:start + batch_size]
        inputs = build_prompts(tokenizer, batch["prompt"])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(eval_cfg["max_new_tokens"]),
                do_sample=False,
                num_beams=int(eval_cfg.get("num_beams", 1)),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        normalized_titles = [normalize_title(text) or "unknown" for text in generated_texts]
        pred_embeddings = pooled_embeddings(
            model=model,
            tokenizer=tokenizer,
            texts=normalized_titles,
            batch_size=max(1, min(batch_size, int(eval_cfg["embedding_batch_size"]))),
            max_length=int(eval_cfg["title_max_length"]),
            show_progress=False,
        )
        pred_embeddings = F.normalize(pred_embeddings, dim=1)
        scores = pred_embeddings @ item_embeddings.T
        ranked = torch.topk(scores, k=max(k_values), dim=1).indices.cpu()

        all_ranked.append(ranked)
        target_indices = [item_to_index[item_id] for item_id in batch["target_item_id"]]
        all_targets.extend(target_indices)

        for row_idx, predicted_title in enumerate(normalized_titles):
            target_title = batch["target_title"][row_idx]
            if predicted_title.lower() == normalize_title(target_title).lower():
                exact_match_hits += 1
            top_indices = ranked[row_idx, :5].tolist()
            top_items = [
                {
                    "item_id": item_ids[item_idx],
                    "title": title_texts[item_idx],
                    "score": round(float(scores[row_idx, item_idx].item()), 6),
                }
                for item_idx in top_indices
            ]
            prediction_rows.append(
                {
                    "user_id": batch["user_id"][row_idx],
                    "target_item_id": batch["target_item_id"][row_idx],
                    "target_title": target_title,
                    "predicted_title": predicted_title,
                    "exact_title_match": predicted_title.lower() == normalize_title(target_title).lower(),
                    "top_5": top_items,
                }
            )

    ranked_indices = torch.cat(all_ranked, dim=0)
    metric_values = rank_metrics(ranked_indices, all_targets, k_values)
    metric_values["exact_title_match"] = exact_match_hits / len(dataset)
    metric_values["num_test_samples"] = len(dataset)
    metric_values["checkpoint_path"] = checkpoint_path
    metric_values["test_path"] = test_path
    metric_values["output_prefix"] = args.output_prefix

    metrics_out = os.path.join(result_dir, f"{args.output_prefix}_metrics.json")
    sample_out = os.path.join(result_dir, f"{args.output_prefix}_sample_predictions.json")
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metric_values, f, indent=2, ensure_ascii=False)
    with open(sample_out, "w", encoding="utf-8") as f:
        json.dump(prediction_rows[:200], f, indent=2, ensure_ascii=False)

    print(json.dumps(metric_values, indent=2))


if __name__ == "__main__":
    main()
