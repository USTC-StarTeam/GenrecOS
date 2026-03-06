#!/usr/bin/env python3

import argparse
import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline_utils import (
    dump_json,
    format_chat_prompt,
    load_item_titles,
    load_jsonl,
    normalize_title,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT model with oracle feedback-based regeneration.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "feedback_regen_config.yaml"),
        help="Path to config yaml.",
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name from config.evaluation.datasets.")
    parser.add_argument("--output_prefix", type=str, default=None, help="Result file prefix.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional checkpoint override.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for debugging.")
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


def build_feedback_prompt(current_prompt: str, prev_title: str, feedback_message: str, round_id: int) -> str:
    return (
        f"{current_prompt}\n\n"
        f"Attempt {round_id} answer: \"{prev_title}\"\n"
        f"{feedback_message}\n"
        "Regenerate the next product title:"
    )


def resolve_dataset_cfg(config: dict, dataset_name: str) -> dict:
    for ds in config["evaluation"]["datasets"]:
        if ds["name"] == dataset_name:
            return ds
    raise ValueError(f"Unknown dataset_name={dataset_name}.")


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths_cfg = config["paths"]
    eval_cfg = config["evaluation"]
    feedback_cfg = config["feedback"]
    ds_cfg = resolve_dataset_cfg(config, args.dataset_name)

    data_dir = os.path.join(script_dir, paths_cfg["data_dir"])
    results_dir = os.path.join(script_dir, paths_cfg["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    output_prefix = args.output_prefix or ds_cfg["output_prefix"]
    checkpoint_path = args.checkpoint_path or os.path.join(script_dir, paths_cfg["checkpoint_path"])
    dataset_path = os.path.join(data_dir, ds_cfg["prepared_file"])
    titles_path = os.path.join(script_dir, paths_cfg["item_titles_path"])

    rows = load_jsonl(dataset_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError(f"Empty dataset: {dataset_path}")

    item_to_title, _ = load_item_titles(titles_path)
    item_ids = list(item_to_title.keys())
    title_texts = [item_to_title[item_id] for item_id in item_ids]
    item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    for row in rows:
        if row["target_item_id"] not in item_to_index:
            raise KeyError(f"target_item_id not found in title pool: {row['target_item_id']}")

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

    batch_size = int(eval_cfg["batch_size"])
    k_values = [int(k) for k in eval_cfg["k_values"]]
    max_k = max(k_values)
    trigger_top_k = int(feedback_cfg["trigger_top_k"])
    max_feedback_rounds = int(feedback_cfg["max_feedback_rounds"])
    sample_trace_limit = int(eval_cfg["sample_trace_limit"])

    num_samples = len(rows)
    target_indices = [item_to_index[row["target_item_id"]] for row in rows]

    baseline_ranked: List[torch.Tensor] = [None] * num_samples
    final_ranked: List[torch.Tensor] = [None] * num_samples
    final_rounds = [0] * num_samples
    final_titles = [""] * num_samples
    exact_match_baseline = 0
    exact_match_final = 0

    sample_attempts = [[] for _ in range(min(sample_trace_limit, num_samples))]
    current_prompts = [row["prompt"] for row in rows]
    active_indices = list(range(num_samples))
    round_stats = []

    for round_id in range(max_feedback_rounds + 1):
        if not active_indices:
            break

        entered = len(active_indices)
        resolved_hit_topk = 0
        forced_stop = 0
        next_active = []

        for start in tqdm(range(0, len(active_indices), batch_size), desc=f"Round {round_id}"):
            batch_indices = active_indices[start:start + batch_size]
            batch_prompts = [current_prompts[idx] for idx in batch_indices]
            inputs = build_prompts(tokenizer, batch_prompts)
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
            ranked = torch.topk(scores, k=max_k, dim=1).indices.cpu()

            for local_idx, sample_idx in enumerate(batch_indices):
                ranked_row = ranked[local_idx]
                pred_title = normalized_titles[local_idx]
                target_idx = target_indices[sample_idx]
                hit_top5 = target_idx in ranked_row[:trigger_top_k].tolist()
                top5_indices = ranked_row[:trigger_top_k].tolist()
                top5_items = [
                    {
                        "item_id": item_ids[item_idx],
                        "title": title_texts[item_idx],
                        "score": round(float(scores[local_idx, item_idx].item()), 6),
                    }
                    for item_idx in top5_indices
                ]

                if round_id == 0:
                    baseline_ranked[sample_idx] = ranked_row
                    if pred_title.lower() == normalize_title(rows[sample_idx]["target_title"]).lower():
                        exact_match_baseline += 1

                if sample_idx < len(sample_attempts):
                    sample_attempts[sample_idx].append(
                        {
                            "round": round_id,
                            "predicted_title": pred_title,
                            "hit_in_top5": hit_top5,
                            "top_5": top5_items,
                        }
                    )

                if hit_top5 or round_id == max_feedback_rounds:
                    final_ranked[sample_idx] = ranked_row
                    final_rounds[sample_idx] = round_id
                    final_titles[sample_idx] = pred_title
                    if pred_title.lower() == normalize_title(rows[sample_idx]["target_title"]).lower():
                        exact_match_final += 1
                    if hit_top5:
                        resolved_hit_topk += 1
                    else:
                        forced_stop += 1
                else:
                    next_active.append(sample_idx)
                    feedback_message = rows[sample_idx].get("feedback_message", feedback_cfg["feedback_message"])
                    current_prompts[sample_idx] = build_feedback_prompt(
                        current_prompt=current_prompts[sample_idx],
                        prev_title=pred_title,
                        feedback_message=feedback_message,
                        round_id=round_id + 1,
                    )

        round_stats.append(
            {
                "round": round_id,
                "entered": entered,
                "resolved_hit_top5": resolved_hit_topk,
                "forced_stop_max_round": forced_stop,
                "remaining": len(next_active),
            }
        )
        active_indices = next_active

    baseline_ranked_tensor = torch.stack(baseline_ranked, dim=0)
    final_ranked_tensor = torch.stack(final_ranked, dim=0)

    baseline_metrics = rank_metrics(baseline_ranked_tensor, target_indices, k_values)
    final_metrics = rank_metrics(final_ranked_tensor, target_indices, k_values)
    baseline_metrics["exact_title_match"] = exact_match_baseline / num_samples
    final_metrics["exact_title_match"] = exact_match_final / num_samples

    baseline_top5_hits = baseline_metrics.get("HR@5", 0.0) * num_samples
    feedback_trigger_count = num_samples - int(round(baseline_top5_hits))
    round_counter = Counter(final_rounds)

    summary = {
        "experiment": "10_feedback_regeneration_oracle",
        "dataset_name": args.dataset_name,
        "num_samples": num_samples,
        "checkpoint_path": checkpoint_path,
        "dataset_path": dataset_path,
        "feedback_trigger_top_k": trigger_top_k,
        "max_feedback_rounds": max_feedback_rounds,
        "baseline_round0_metrics": baseline_metrics,
        "final_after_feedback_metrics": final_metrics,
        "delta_final_minus_baseline": {
            key: final_metrics[key] - baseline_metrics[key]
            for key in baseline_metrics.keys()
            if key in final_metrics and key.startswith(("HR@", "NDCG@", "exact_title_match"))
        },
        "feedback_diagnostics": {
            "feedback_trigger_count": feedback_trigger_count,
            "feedback_trigger_rate": feedback_trigger_count / num_samples,
            "avg_generation_round_index": float(np.mean(final_rounds)),
            "avg_generation_calls_per_sample": float(np.mean([x + 1 for x in final_rounds])),
            "final_round_distribution": {str(k): int(v) for k, v in sorted(round_counter.items())},
            "round_stats": round_stats,
        },
    }

    metrics_path = os.path.join(results_dir, f"{output_prefix}_metrics.json")
    traces = []
    for idx in range(min(sample_trace_limit, num_samples)):
        traces.append(
            {
                "user_id": rows[idx]["user_id"],
                "target_item_id": rows[idx]["target_item_id"],
                "target_title": rows[idx]["target_title"],
                "final_round": final_rounds[idx],
                "final_predicted_title": final_titles[idx],
                "attempts": sample_attempts[idx],
            }
        )
    traces_path = os.path.join(results_dir, f"{output_prefix}_sample_traces.json")

    dump_json(metrics_path, summary)
    dump_json(traces_path, traces)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
