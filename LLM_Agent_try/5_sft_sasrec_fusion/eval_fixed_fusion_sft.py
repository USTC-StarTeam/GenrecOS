#!/usr/bin/env python3

import argparse
import os

import torch

from common import (
    RESULTS_DIR,
    cache_path,
    ensure_dir,
    evaluate_fixed_alpha,
    find_best_fixed_alpha,
    move_split_to_device,
    save_json,
    set_global_seed,
    wait_for_cache_ready,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed fusion with SFT semantic scores.")
    parser.add_argument("--cache_tag", type=str, default="sft_best_full")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=192)
    parser.add_argument("--run_name", type=str, default="fixed_fusion_sft")
    parser.add_argument("--poll_seconds", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(RESULTS_DIR)
    set_global_seed(args.seed)

    cache_meta = wait_for_cache_ready(args.cache_tag, args.poll_seconds)
    print(f"Cache ready, loading from {args.cache_tag}: {cache_meta['split_sizes']}")

    device = torch.device(args.device)
    val_split = torch.load(cache_path(args.cache_tag, "val_features.pt"), map_location="cpu", weights_only=False)
    test_split = torch.load(cache_path(args.cache_tag, "test_features.pt"), map_location="cpu", weights_only=False)

    val_split = move_split_to_device(val_split, device)
    test_split = move_split_to_device(test_split, device)

    best_alpha, val_metrics = find_best_fixed_alpha(
        val_split["sas_scores"],
        val_split["llm_scores"],
        val_split["targets"],
        args.eval_batch_size,
    )
    test_fixed = evaluate_fixed_alpha(
        test_split["sas_scores"],
        test_split["llm_scores"],
        test_split["targets"],
        best_alpha,
        args.eval_batch_size,
    )
    test_sas = evaluate_fixed_alpha(
        test_split["sas_scores"],
        test_split["llm_scores"],
        test_split["targets"],
        1.0,
        args.eval_batch_size,
    )
    test_sft = evaluate_fixed_alpha(
        test_split["sas_scores"],
        test_split["llm_scores"],
        test_split["targets"],
        0.0,
        args.eval_batch_size,
    )

    results = {
        "cache_tag": args.cache_tag,
        "run_name": args.run_name,
        "split_sizes": cache_meta["split_sizes"],
        "best_alpha": best_alpha,
        "val_fixed_metrics": val_metrics,
        "test": {
            "sasrec": test_sas,
            "sft_semantic": test_sft,
            "fixed_fusion": test_fixed,
        },
    }

    output_path = os.path.join(RESULTS_DIR, f"{args.run_name}_results.json")
    save_json(output_path, results)
    print(f"Saved results to {output_path}")
    print(results)


if __name__ == "__main__":
    main()
