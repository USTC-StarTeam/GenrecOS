#!/usr/bin/env python3

import argparse
import gc
import json
import os

import torch

from common import (
    DEFAULT_MAX_PROMPT_LENGTH,
    TITLE_MAX_LENGTH,
    compute_metrics_from_scores,
    compute_sasrec_scores,
    encode_item_titles,
    evaluation_result_path,
    find_best_alpha,
    generate_predictions_and_scores,
    get_track_best_model_dir,
    load_causal_model,
    get_track_dir,
    load_item_mapping,
    load_ordered_titles,
    load_sasrec_model,
    load_jsonl,
    load_json,
    save_json,
    set_global_seed,
    summarize_tool_behavior,
    teacher_summary_path,
    evaluate_fixed_alpha,
    evaluate_routed_alpha,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tool-aware routing + SASRec fusion.")
    parser.add_argument("--track_name", type=str, choices=["pre_sft", "post_sft"], required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title_batch_size", type=int, default=128)
    parser.add_argument("--predict_batch_size", type=int, default=16)
    parser.add_argument("--sasrec_batch_size", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    return parser.parse_args()


def evaluate_split(rows, llm_scores, sas_scores, targets, route_mask):
    semantic_metrics = compute_metrics_from_scores(llm_scores.float(), targets, [1, 5, 10, 20])
    best_fixed_alpha, best_fixed_metrics = find_best_alpha(sas_scores, llm_scores, targets, route_mask=None)
    best_route_alpha, best_route_metrics = find_best_alpha(sas_scores, llm_scores, targets, route_mask=route_mask)
    return {
        "semantic_only": semantic_metrics,
        "split_best_fixed_alpha": best_fixed_alpha,
        "split_best_fixed_metrics": best_fixed_metrics,
        "split_best_routed_alpha": best_route_alpha,
        "split_best_routed_metrics": best_route_metrics,
    }


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    item_to_id, _ = load_item_mapping()
    _, ordered_titles = load_ordered_titles(item_to_id)
    model_path = get_track_best_model_dir(args.track_name)
    teacher_summary = load_json(teacher_summary_path(args.track_name))

    eval_model, eval_tokenizer = load_causal_model(
        model_path=model_path,
        device=device,
        add_tool_token=False,
        train_mode=False,
    )
    sasrec_model, sasrec_tokenizer = load_sasrec_model(device)
    item_embeddings = encode_item_titles(
        model=eval_model,
        tokenizer=eval_tokenizer,
        ordered_titles=ordered_titles,
        batch_size=args.title_batch_size,
        device=device,
        max_length=TITLE_MAX_LENGTH,
    )

    results = {
        "track_name": args.track_name,
        "checkpoint_path": model_path,
        "teacher_summary": teacher_summary,
        "splits": {},
    }

    best_route_alpha = None
    best_fixed_alpha = None
    for split_name in ["val", "test"]:
        rows = load_jsonl(os.path.join(get_track_dir(args.track_name), f"{split_name}.jsonl"))
        generated = generate_predictions_and_scores(
            model=eval_model,
            tokenizer=eval_tokenizer,
            rows=rows,
            item_embeddings=item_embeddings,
            batch_size=args.predict_batch_size,
            device=device,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            prompt_style="tool",
        )
        sas_scores = compute_sasrec_scores(
            sasrec_model=sasrec_model,
            sasrec_tokenizer=sasrec_tokenizer,
            histories=[row["sasrec_history_ids"] for row in rows],
            batch_size=args.sasrec_batch_size,
            device=device,
        )
        targets = torch.tensor([row["target_internal_id"] for row in rows], dtype=torch.long)
        route_mask = torch.tensor(generated["tool_flags"], dtype=torch.bool)
        metrics = evaluate_split(rows, generated["llm_scores"], sas_scores, targets, route_mask)
        behavior = summarize_tool_behavior(
            generated_flags=generated["tool_flags"],
            teacher_flags=[bool(row["tool_label"]) for row in rows],
        )
        results["splits"][split_name] = {
            "num_rows": len(rows),
            "metrics": metrics,
            "tool_behavior": behavior,
            "sample_predictions": [
                {
                    "user_id": rows[idx]["user_id"],
                    "target_title": rows[idx]["target_title"],
                    "raw_prediction": generated["raw_predictions"][idx],
                    "stripped_title": generated["stripped_titles"][idx],
                    "triggered_tool": bool(generated["tool_flags"][idx]),
                    "teacher_tool_label": bool(rows[idx]["tool_label"]),
                }
                for idx in range(min(100, len(rows)))
            ],
        }
        if split_name == "val":
            best_route_alpha = metrics["split_best_routed_alpha"]
            best_fixed_alpha = metrics["split_best_fixed_alpha"]
            results["splits"][split_name]["metrics"]["used_fixed_alpha"] = best_fixed_alpha
            results["splits"][split_name]["metrics"]["used_fixed_metrics"] = metrics["split_best_fixed_metrics"]
            results["splits"][split_name]["metrics"]["used_routed_alpha"] = best_route_alpha
            results["splits"][split_name]["metrics"]["used_routed_metrics"] = metrics["split_best_routed_metrics"]
        else:
            results["splits"][split_name]["metrics"]["used_fixed_alpha"] = best_fixed_alpha
            results["splits"][split_name]["metrics"]["used_fixed_metrics"] = evaluate_fixed_alpha(
                sas_scores, generated["llm_scores"], targets, best_fixed_alpha
            )
            results["splits"][split_name]["metrics"]["used_routed_alpha"] = best_route_alpha
            results["splits"][split_name]["metrics"]["used_routed_metrics"] = evaluate_routed_alpha(
                sas_scores, generated["llm_scores"], targets, route_mask, best_route_alpha
            )
            results["best_fixed_alpha"] = best_fixed_alpha
            results["best_routed_alpha"] = best_route_alpha

    save_json(evaluation_result_path(args.track_name), results)
    print(json.dumps(results["splits"]["test"]["metrics"], indent=2, ensure_ascii=False))

    del eval_model, sasrec_model, item_embeddings
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
