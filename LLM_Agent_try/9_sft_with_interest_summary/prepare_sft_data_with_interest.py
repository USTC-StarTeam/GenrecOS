#!/usr/bin/env python3

import argparse
import asyncio
import copy
import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import yaml
from openai import AsyncOpenAI
from tqdm import tqdm

from pipeline_utils import (
    build_prompt,
    dump_json,
    dump_jsonl,
    ensure_dir,
    load_item_titles,
    parse_raw_sequences,
    summarize_rows,
    truncate_review,
)

DEFAULT_UNAVAILABLE = "- unavailable summary\n- unavailable summary\n- unavailable summary"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SFT data with external interest summary for train/val/test.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sft_interest_config.yaml"),
    )
    parser.add_argument(
        "--max_users",
        type=int,
        default=0,
        help="For smoke test only. If >0, keep at most this many users.",
    )
    return parser.parse_args()


def load_item_metadata(meta_path: str) -> Dict[str, dict]:
    meta = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = row.get("parent_asin")
            if not item_id:
                continue
            meta[item_id] = {
                "title": row.get("title"),
                "store": row.get("store"),
                "main_category": row.get("main_category"),
                "categories": row.get("categories") or [],
                "features": row.get("features") or [],
                "description": row.get("description") or [],
                "details": row.get("details") or {},
                "average_rating": row.get("average_rating"),
                "rating_number": row.get("rating_number"),
                "price": row.get("price"),
            }
    return meta


def compact_text(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_metadata_snippet(meta: dict, max_chars: int) -> str:
    if not meta:
        return "metadata: N/A"
    categories = " > ".join(meta.get("categories", [])[:5])
    features = " ; ".join(meta.get("features", [])[:4])
    desc = " ".join(meta.get("description", [])[:2])
    details = meta.get("details", {})
    detail_items = list(details.items())[:6] if isinstance(details, dict) else []
    details_text = " ; ".join(f"{k}: {v}" for k, v in detail_items)
    fields = [
        f"meta_title={meta.get('title') or 'N/A'}",
        f"store={meta.get('store') or 'N/A'}",
        f"main_category={meta.get('main_category') or 'N/A'}",
        f"avg_rating={meta.get('average_rating')}",
        f"rating_number={meta.get('rating_number')}",
        f"price={meta.get('price')}",
        f"categories={categories or 'N/A'}",
        f"features={features or 'N/A'}",
        f"description={desc or 'N/A'}",
        f"details={details_text or 'N/A'}",
    ]
    return compact_text(" | ".join(fields), max_chars)


def make_row(
    split: str,
    user_id: str,
    history: List[Tuple[str, float, str]],
    target_item_id: str,
    item_to_title: Dict[str, str],
    metadata_map: Dict[str, dict],
    max_review_len: int,
    summary_meta_char_limit: int,
) -> dict:
    history_item_ids = [item_id for item_id, _, _ in history]
    history_titles = [item_to_title[item_id] for item_id, _, _ in history]
    history_ratings = [int(rating) for _, rating, _ in history]
    history_reviews_full = [review or "" for _, _, review in history]
    history_reviews = [truncate_review(review or "", max_review_len) for _, _, review in history]
    history_metadata = [
        format_metadata_snippet(metadata_map.get(item_id), summary_meta_char_limit)
        for item_id in history_item_ids
    ]
    prompt_original = build_prompt(history_titles, history_ratings, history_reviews)
    return {
        "split": split,
        "user_id": user_id,
        "history_item_ids": history_item_ids,
        "history_titles": history_titles,
        "history_ratings": history_ratings,
        "history_reviews": history_reviews,
        "history_reviews_full": history_reviews_full,
        "history_item_metadata": history_metadata,
        "target_item_id": target_item_id,
        "target_title": item_to_title[target_item_id],
        "prompt_original": prompt_original,
        "prompt": prompt_original,
    }


def build_split_rows(
    user_sequences: Dict[str, List[Tuple[str, int, float, str]]],
    item_to_title: Dict[str, str],
    metadata_map: Dict[str, dict],
    max_history: int,
    max_review_len: int,
    summary_meta_char_limit: int,
) -> Dict[str, List[dict]]:
    rows_by_split = defaultdict(list)
    for user_id, sequence in user_sequences.items():
        if len(sequence) < 3:
            continue
        items = [(item_id, rating, review_text) for item_id, _, rating, review_text in sequence]

        test_history = items[:-1][-max_history:]
        rows_by_split["test"].append(
            make_row(
                split="test",
                user_id=user_id,
                history=test_history,
                target_item_id=items[-1][0],
                item_to_title=item_to_title,
                metadata_map=metadata_map,
                max_review_len=max_review_len,
                summary_meta_char_limit=summary_meta_char_limit,
            )
        )

        val_history = items[:-2][-max_history:]
        rows_by_split["val"].append(
            make_row(
                split="val",
                user_id=user_id,
                history=val_history,
                target_item_id=items[-2][0],
                item_to_title=item_to_title,
                metadata_map=metadata_map,
                max_review_len=max_review_len,
                summary_meta_char_limit=summary_meta_char_limit,
            )
        )

        train_items = items[:-2]
        for target_idx in range(1, len(train_items)):
            history = train_items[:target_idx][-max_history:]
            rows_by_split["train"].append(
                make_row(
                    split="train",
                    user_id=user_id,
                    history=history,
                    target_item_id=train_items[target_idx][0],
                    item_to_title=item_to_title,
                    metadata_map=metadata_map,
                    max_review_len=max_review_len,
                    summary_meta_char_limit=summary_meta_char_limit,
                )
            )
    return rows_by_split


def summary_key(row: dict) -> str:
    payload = {
        "split": row["split"],
        "user_id": row["user_id"],
        "history_item_ids": row["history_item_ids"],
        "history_ratings": row["history_ratings"],
        "history_reviews_full": row["history_reviews_full"],
        "target_item_id": row["target_item_id"],
    }
    digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{row['split']}::{row['user_id']}::{digest}"


def build_summary_user_prompt(row: dict, summary_review_char_limit: int, summary_meta_char_limit: int) -> str:
    lines = []
    for idx, title in enumerate(row["history_titles"], 1):
        rating = row["history_ratings"][idx - 1] if idx - 1 < len(row["history_ratings"]) else "NA"
        review = row["history_reviews_full"][idx - 1] if idx - 1 < len(row["history_reviews_full"]) else ""
        meta = row["history_item_metadata"][idx - 1] if idx - 1 < len(row["history_item_metadata"]) else "N/A"
        lines.append(
            f"{idx}. title={title}\n"
            f"   rating={rating}\n"
            f"   review={compact_text(review, summary_review_char_limit)}\n"
            f"   item_metadata={compact_text(meta, summary_meta_char_limit)}"
        )
    history_block = "\n".join(lines)
    return (
        "You are an expert recommendation analyst.\n"
        "Given a user's purchase history (with review text and item metadata), infer stable preference signals.\n"
        "Return exactly 4 bullet points.\n"
        "Each bullet should be concrete and recommendation-oriented.\n"
        "Do not output any explanation outside bullet points.\n\n"
        f"User split: {row['split']}\n"
        f"History:\n{history_block}\n"
    )


def canonicalize_summary_text(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return "\n".join(lines)


def is_invalid_summary(text: str) -> bool:
    normalized = canonicalize_summary_text(text).lower()
    if not normalized:
        return True
    if "unavailable summary" in normalized:
        return True
    return False


def extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(block, str) and block.strip():
                parts.append(block.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


async def summarize_one(client: AsyncOpenAI, row: dict, cfg: dict, data_cfg: dict, sem: asyncio.Semaphore) -> Tuple[str, str, str]:
    key = summary_key(row)
    prompt = build_summary_user_prompt(
        row=row,
        summary_review_char_limit=int(data_cfg["summary_review_char_limit"]),
        summary_meta_char_limit=int(data_cfg["summary_meta_char_limit"]),
    )
    max_retries = int(cfg["max_retries"])
    retry_base_seconds = float(cfg["retry_base_seconds"])

    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=cfg["model"],
                    messages=[
                        {"role": "system", "content": "You summarize user interests for recommendation training."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=float(cfg["temperature"]),
                    max_tokens=int(cfg["max_tokens"]),
                    timeout=int(cfg["timeout_seconds"]),
                )
                text = extract_text_from_content(response.choices[0].message.content)
                if text:
                    return key, text, "ok"
                if attempt == max_retries - 1:
                    return key, DEFAULT_UNAVAILABLE, "empty"
            except Exception:
                if attempt == max_retries - 1:
                    return key, DEFAULT_UNAVAILABLE, "error"
            await asyncio.sleep(retry_base_seconds * (2 ** attempt))
    return key, DEFAULT_UNAVAILABLE, "error"


async def build_summaries(
    all_rows: List[dict],
    cache: Dict[str, str],
    cfg: dict,
    data_cfg: dict,
    cache_path: str,
) -> Tuple[Dict[str, str], dict]:
    api_base_url = cfg["api_base_url"]
    if api_base_url.endswith("/chat/completions"):
        api_base_url = api_base_url[: -len("/chat/completions")]
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=api_base_url)
    sem = asyncio.Semaphore(int(cfg["concurrency"]))
    refresh_invalid_cache = bool(cfg.get("refresh_invalid_cache", True))

    pending_rows = []
    reused_count = 0
    refreshed_invalid_count = 0
    for row in all_rows:
        key = summary_key(row)
        cached = cache.get(key)
        if cached is None:
            pending_rows.append(row)
            continue
        if refresh_invalid_cache and is_invalid_summary(cached):
            pending_rows.append(row)
            refreshed_invalid_count += 1
            continue
        reused_count += 1

    stats = {
        "total_rows": len(all_rows),
        "pending_rows": len(pending_rows),
        "reused_cache_count": reused_count,
        "refreshed_invalid_cache_count": refreshed_invalid_count,
        "ok_count": 0,
        "empty_count": 0,
        "error_count": 0,
    }

    if not pending_rows:
        return cache, stats

    tasks = [asyncio.create_task(summarize_one(client, row, cfg, data_cfg, sem)) for row in pending_rows]
    completed = 0
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Summarizing train/val/test interests"):
        key, text, tag = await fut
        cache[key] = text
        stats[f"{tag}_count"] += 1
        completed += 1
        if completed % 20 == 0:
            ensure_dir(os.path.dirname(cache_path))
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

    ensure_dir(os.path.dirname(cache_path))
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    return cache, stats


def augment_prompt(original_prompt: str, summary_text: str) -> str:
    marker = "Based on the user's interaction history, predict the next product they would be most interested in purchasing."
    block = (
        "Inferred user interests from purchase history + review text + item metadata:\n"
        f"{summary_text}\n\n"
    )
    if marker in original_prompt:
        return original_prompt.replace(marker, block + marker)
    return original_prompt + "\n\n" + block


def align_common_4385(rows: List[dict], sasrec_test_path: str, item_mapping_path: str) -> List[dict]:
    sasrec_test = json.load(open(sasrec_test_path, "r", encoding="utf-8"))
    mapping = json.load(open(item_mapping_path, "r", encoding="utf-8"))
    item_to_id = mapping["item_to_id"]
    sasrec_pairs = {(row["user_id"], int(row["ground_truth"])) for row in sasrec_test}

    kept = []
    for row in rows:
        internal_id = item_to_id.get(row["target_item_id"])
        if internal_id is None:
            continue
        if (row["user_id"], int(internal_id)) not in sasrec_pairs:
            continue
        kept.append(row)
    return kept


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    data_cfg = config["data"]
    summary_cfg = config["interest_summary"]

    data_dir = os.path.join(script_dir, paths["data_dir"])
    cache_dir = os.path.join(script_dir, paths["cache_dir"])
    ensure_dir(data_dir)
    ensure_dir(cache_dir)

    item_titles_path = os.path.join(script_dir, paths["item_titles_path"])
    raw_data_path = os.path.join(script_dir, paths["raw_data_path"])
    meta_data_path = os.path.join(script_dir, paths["meta_data_path"])
    sasrec_test_path = os.path.join(script_dir, paths["sasrec_test_path"])
    item_mapping_path = os.path.join(script_dir, paths["item_mapping_path"])

    item_to_title, _ = load_item_titles(item_titles_path)
    metadata_map = load_item_metadata(meta_data_path)
    user_sequences = parse_raw_sequences(
        raw_data_path=raw_data_path,
        item_to_title=item_to_title,
        min_item_freq=int(data_cfg["min_item_freq"]),
        min_seq_length=int(data_cfg["min_seq_length"]),
    )
    if args.max_users and args.max_users > 0:
        kept = dict(list(user_sequences.items())[: int(args.max_users)])
        user_sequences = kept

    rows_by_split = build_split_rows(
        user_sequences=user_sequences,
        item_to_title=item_to_title,
        metadata_map=metadata_map,
        max_history=int(data_cfg["max_history"]),
        max_review_len=int(data_cfg["max_review_len"]),
        summary_meta_char_limit=int(data_cfg["summary_meta_char_limit"]),
    )

    raw_rows_by_split = copy.deepcopy(rows_by_split)

    for split in ("train", "val", "test"):
        dump_jsonl(os.path.join(data_dir, f"{split}_raw_fullfields.jsonl"), raw_rows_by_split[split])

    cache_path = os.path.join(cache_dir, summary_cfg["cache_file"])
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, "r", encoding="utf-8"))
    else:
        cache = {}

    all_rows = rows_by_split["train"] + rows_by_split["val"] + rows_by_split["test"]
    cache, summary_stats = asyncio.run(build_summaries(all_rows, cache, summary_cfg, data_cfg, cache_path))

    for split in ("train", "val", "test"):
        for row in rows_by_split[split]:
            key = summary_key(row)
            summary_text = cache.get(key, DEFAULT_UNAVAILABLE)
            row["interest_summary"] = summary_text
            row["prompt"] = augment_prompt(row["prompt_original"], summary_text)

    def to_training_row(row: dict) -> dict:
        # Drop oversized debug fields before SFT tokenization.
        return {
            "split": row["split"],
            "user_id": row["user_id"],
            "history_item_ids": row["history_item_ids"],
            "history_titles": row["history_titles"],
            "history_ratings": row["history_ratings"],
            "history_reviews": row["history_reviews"],
            "target_item_id": row["target_item_id"],
            "target_title": row["target_title"],
            "prompt_original": row["prompt_original"],
            "interest_summary": row.get("interest_summary", DEFAULT_UNAVAILABLE),
            "prompt": row["prompt"],
        }

    raw_rows_trimmed = {split: [to_training_row(row) for row in raw_rows_by_split[split]] for split in ("train", "val", "test")}
    rows_trimmed = {split: [to_training_row(row) for row in rows_by_split[split]] for split in ("train", "val", "test")}

    for split in ("train", "val", "test"):
        dump_jsonl(os.path.join(data_dir, f"{split}_raw.jsonl"), raw_rows_trimmed[split])

    # train_full_sft_interest.py reads train.jsonl / val.jsonl / test.jsonl.
    for split in ("train", "val", "test"):
        dump_jsonl(os.path.join(data_dir, f"{split}.jsonl"), rows_trimmed[split])

    test_common_raw = align_common_4385(raw_rows_trimmed["test"], sasrec_test_path, item_mapping_path)
    dump_jsonl(os.path.join(data_dir, "test_raw_common_4385.jsonl"), test_common_raw)
    test_common_interest = align_common_4385(rows_trimmed["test"], sasrec_test_path, item_mapping_path)
    dump_jsonl(os.path.join(data_dir, "test_common_4385.jsonl"), test_common_interest)

    summary = {
        "task": "sft_with_interest_summary_all_splits",
        "data_config": data_cfg,
        "summary_model": summary_cfg["model"],
        "num_users_after_filtering": len(user_sequences),
        "metadata_items_loaded": len(metadata_map),
        "train_raw": summarize_rows(raw_rows_trimmed["train"]),
        "val_raw": summarize_rows(raw_rows_trimmed["val"]),
        "test_raw": summarize_rows(raw_rows_trimmed["test"]),
        "test_common_4385_size": len(test_common_interest),
        "cache_path": cache_path,
        "cache_size": len(cache),
        "summary_generation_stats": summary_stats,
    }
    dump_json(os.path.join(data_dir, "dataset_summary_with_interest.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
