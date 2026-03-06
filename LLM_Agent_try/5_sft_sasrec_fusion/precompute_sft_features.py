#!/usr/bin/env python3

import argparse
import gc
import os

import torch

from common import (
    RESULTS_DIR,
    cache_path,
    ensure_dir,
    get_cache_dir,
    load_all_splits,
    load_item_mapping,
    load_item_titles,
    load_sasrec_model,
    load_sft_model,
    precompute_split_features,
    save_json,
    set_global_seed,
    success_path,
    encode_item_titles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute SASRec + SFT fusion features.")
    parser.add_argument("--cache_tag", type=str, default="sft_best_full")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title_batch_size", type=int, default=128)
    parser.add_argument("--precompute_batch_size", type=int, default=24)
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_val", type=int, default=0)
    parser.add_argument("--limit_test", type=int, default=0)
    parser.add_argument("--force_recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(RESULTS_DIR)
    ensure_dir(get_cache_dir(args.cache_tag))
    set_global_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    item_to_id, _ = load_item_mapping()
    ordered_item_ids, ordered_titles = load_item_titles(item_to_id)
    splits = load_all_splits(item_to_id, args.limit_train, args.limit_val, args.limit_test)

    print("Aligned split sizes:")
    for split_name, split in splits.items():
        print(f"  {split_name}: {len(split.prompts)}")

    if os.path.exists(success_path(args.cache_tag)) and not args.force_recompute:
        print(f"Cache already ready: {success_path(args.cache_tag)}")
        return

    sasrec_model, sasrec_tokenizer = load_sasrec_model(device)
    sft_model, sft_tokenizer = load_sft_model(device)

    print("Encoding SFT catalog embeddings...")
    item_embeddings = encode_item_titles(
        model=sft_model,
        tokenizer=sft_tokenizer,
        ordered_titles=ordered_titles,
        batch_size=args.title_batch_size,
        device=device,
    )
    torch.save(
        {
            "item_ids": ordered_item_ids,
            "item_embeddings": item_embeddings,
        },
        cache_path(args.cache_tag, "item_embeddings.pt"),
    )

    split_sizes = {}
    for split_name, split_data in splits.items():
        print(f"Precomputing split: {split_name}")
        features = precompute_split_features(
            split_name=split_name,
            split_data=split_data,
            sasrec_model=sasrec_model,
            sasrec_tokenizer=sasrec_tokenizer,
            sft_model=sft_model,
            sft_tokenizer=sft_tokenizer,
            item_embeddings=item_embeddings,
            device=device,
            batch_size=args.precompute_batch_size,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
        )
        torch.save(features, cache_path(args.cache_tag, f"{split_name}_features.pt"))
        split_sizes[split_name] = len(split_data.prompts)

    del sasrec_model, sft_model, item_embeddings
    gc.collect()
    torch.cuda.empty_cache()

    save_json(
        success_path(args.cache_tag),
        {
            "cache_tag": args.cache_tag,
            "split_sizes": split_sizes,
            "device": str(device),
            "title_batch_size": args.title_batch_size,
            "precompute_batch_size": args.precompute_batch_size,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
        },
    )
    print(f"Cache ready: {success_path(args.cache_tag)}")


if __name__ == "__main__":
    main()
