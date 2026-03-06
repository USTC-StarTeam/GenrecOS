#!/usr/bin/env python3

import argparse
import gc
import os

import torch

from common import (
    CACHE_DIR,
    DATA_DIR,
    DEFAULT_MAX_PROMPT_LENGTH,
    RESULTS_DIR,
    compute_label_summary,
    compute_metrics_from_scores,
    compute_sasrec_scores,
    encode_item_titles,
    ensure_dir,
    generate_predictions_and_scores,
    get_track_dir,
    load_aligned_rows,
    load_causal_model,
    load_item_mapping,
    load_ordered_titles,
    load_sasrec_model,
    save_json,
    save_jsonl,
    set_global_seed,
    teacher_summary_path,
    choose_model_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tool-SFT datasets using model-vs-SASRec correctness.")
    parser.add_argument("--track_name", type=str, choices=["pre_sft", "post_sft"], required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title_batch_size", type=int, default=128)
    parser.add_argument("--predict_batch_size", type=int, default=16)
    parser.add_argument("--sasrec_batch_size", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--limit_per_split", type=int, default=0)
    parser.add_argument("--force_recompute", action="store_true")
    return parser.parse_args()


def build_split_rows(
    rows: list,
    llm_scores: torch.Tensor,
    sas_scores: torch.Tensor,
    raw_predictions: list,
    stripped_titles: list,
    targets: torch.Tensor,
) -> tuple[list, dict]:
    summary = compute_label_summary(llm_scores, sas_scores, targets)
    tool_labels = summary["tool_labels"].tolist()
    llm_hit1 = summary["llm_hit1"].tolist()
    sas_hit1 = summary["sas_hit1"].tolist()
    llm_top1 = summary["llm_top1"].tolist()
    sas_top1 = summary["sas_top1"].tolist()

    output_rows = []
    for idx, row in enumerate(rows):
        assistant_target = row["target_title"]
        if tool_labels[idx]:
            assistant_target = f"[tool:seqscore] {assistant_target}"
        new_row = dict(row)
        new_row["assistant_target"] = assistant_target
        new_row["teacher_raw_prediction"] = raw_predictions[idx]
        new_row["teacher_predicted_title"] = stripped_titles[idx]
        new_row["teacher_llm_top1_internal"] = int(llm_top1[idx])
        new_row["teacher_sasrec_top1_internal"] = int(sas_top1[idx])
        new_row["teacher_llm_hit1"] = bool(llm_hit1[idx])
        new_row["teacher_sasrec_hit1"] = bool(sas_hit1[idx])
        new_row["tool_label"] = bool(tool_labels[idx])
        output_rows.append(new_row)

    split_summary = {
        "num_rows": len(rows),
        "tool_positive_rate": sum(tool_labels) / len(tool_labels) if tool_labels else 0.0,
        "teacher_llm_metrics": compute_metrics_from_scores(llm_scores.float(), targets, [1, 5, 10, 20]),
        "teacher_sasrec_metrics": compute_metrics_from_scores(sas_scores.float(), targets, [1, 5, 10, 20]),
    }
    return output_rows, split_summary


def main() -> None:
    args = parse_args()
    ensure_dir(DATA_DIR)
    ensure_dir(RESULTS_DIR)
    ensure_dir(CACHE_DIR)
    set_global_seed(args.seed)

    output_dir = get_track_dir(args.track_name)
    summary_path = teacher_summary_path(args.track_name)
    if os.path.exists(summary_path) and not args.force_recompute:
        print(f"Teacher summary already exists: {summary_path}")
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    model_path = choose_model_path(args.track_name)
    item_to_id, _ = load_item_mapping()
    ordered_item_ids, ordered_titles = load_ordered_titles(item_to_id)

    split_rows = {
        "train": load_aligned_rows("train", item_to_id, args.limit_per_split),
        "val": load_aligned_rows("val", item_to_id, args.limit_per_split),
        "test": load_aligned_rows("test", item_to_id, args.limit_per_split),
    }

    teacher_model, teacher_tokenizer = load_causal_model(
        model_path=model_path,
        device=device,
        add_tool_token=False,
        train_mode=False,
    )
    sasrec_model, sasrec_tokenizer = load_sasrec_model(device)

    item_embeddings = encode_item_titles(
        model=teacher_model,
        tokenizer=teacher_tokenizer,
        ordered_titles=ordered_titles,
        batch_size=args.title_batch_size,
        device=device,
    )

    track_summary = {
        "track_name": args.track_name,
        "teacher_model_path": model_path,
        "tool_rule": "tool_label = (llm_top1_wrong and sasrec_top1_correct)",
        "split_sizes": {},
        "splits": {},
    }

    for split_name, rows in split_rows.items():
        print(f"Preparing split: {split_name} ({len(rows)} rows)")
        teacher_outputs = generate_predictions_and_scores(
            model=teacher_model,
            tokenizer=teacher_tokenizer,
            rows=rows,
            item_embeddings=item_embeddings,
            batch_size=args.predict_batch_size,
            device=device,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            prompt_style="original",
        )
        sas_scores = compute_sasrec_scores(
            sasrec_model=sasrec_model,
            sasrec_tokenizer=sasrec_tokenizer,
            histories=[row["sasrec_history_ids"] for row in rows],
            batch_size=args.sasrec_batch_size,
            device=device,
        )
        targets = torch.tensor([row["target_internal_id"] for row in rows], dtype=torch.long)
        output_rows, split_summary = build_split_rows(
            rows=rows,
            llm_scores=teacher_outputs["llm_scores"],
            sas_scores=sas_scores,
            raw_predictions=teacher_outputs["raw_predictions"],
            stripped_titles=teacher_outputs["stripped_titles"],
            targets=targets,
        )
        save_jsonl(os.path.join(output_dir, f"{split_name}.jsonl"), output_rows)
        track_summary["split_sizes"][split_name] = len(rows)
        track_summary["splits"][split_name] = split_summary

    save_json(summary_path, track_summary)
    print(f"Saved tool-SFT data and teacher summary to {output_dir}")

    del teacher_model, sasrec_model, item_embeddings
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
