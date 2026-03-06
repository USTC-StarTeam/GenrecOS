#!/usr/bin/env python3

import argparse
import asyncio
import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

import yaml
from openai import AsyncOpenAI
from tqdm import tqdm

from common import dump_jsonl, ensure_dir, load_jsonl

DEFAULT_NO_PREFERENCE = "- no clear preference\n- no clear preference\n- no clear preference"
DEFAULT_UNAVAILABLE = "- unavailable summary\n- unavailable summary\n- unavailable summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare interest-summary augmented test data.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "experiment_config.yaml"),
    )
    return parser.parse_args()


def summary_key(row: dict) -> str:
    payload = {
        "user_id": row["user_id"],
        "history_item_ids": row.get("history_item_ids", []),
        "history_titles": row.get("history_titles", []),
        "history_ratings": row.get("history_ratings", []),
        "history_reviews": row.get("history_reviews", []),
    }
    digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{row['user_id']}::{digest}"


def compact_review(text: str, max_chars: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_summary_user_prompt(row: dict, max_history_items: int, review_char_limit: int) -> str:
    titles = row.get("history_titles", [])[-max_history_items:]
    ratings = row.get("history_ratings", [])[-max_history_items:]
    reviews = row.get("history_reviews", [])[-max_history_items:]
    lines = []
    for i, title in enumerate(titles):
        rating = ratings[i] if i < len(ratings) else "NA"
        review = compact_review(reviews[i] if i < len(reviews) else "", review_char_limit)
        lines.append(f"{i+1}. title={title} | rating={rating} | review={review}")
    history_text = "\n".join(lines)
    return (
        "You are given a user's historical purchases.\n"
        "Summarize the user's stable shopping interests and preference signals for next-item recommendation.\n"
        "Output exactly 3 bullet points, concise, no extra explanation.\n\n"
        f"History:\n{history_text}"
    )


def canonicalize_summary_text(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return "\n".join(lines)


def is_placeholder_summary(text: str) -> bool:
    canonical = canonicalize_summary_text(text).lower()
    return canonical in {
        canonicalize_summary_text(DEFAULT_NO_PREFERENCE).lower(),
        canonicalize_summary_text(DEFAULT_UNAVAILABLE).lower(),
    }


def extract_text_from_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                part = block.get("text") or block.get("content") or ""
                if isinstance(part, str) and part.strip():
                    parts.append(part.strip())
            elif isinstance(block, str) and block.strip():
                parts.append(block.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


async def request_summary_once(
    client: AsyncOpenAI,
    model_name: str,
    prompt: str,
    cfg: dict,
) -> Tuple[str, str]:
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You summarize user interests for recommendation."},
            {"role": "user", "content": prompt},
        ],
        temperature=float(cfg["temperature"]),
        max_tokens=int(cfg["max_tokens"]),
        timeout=int(cfg["timeout_seconds"]),
    )
    finish_reason = response.choices[0].finish_reason or "unknown"
    text = extract_text_from_message_content(response.choices[0].message.content)
    return text, finish_reason


async def summarize_one(
    client: AsyncOpenAI,
    row: dict,
    cfg: dict,
    sem: asyncio.Semaphore,
) -> Tuple[str, str, str]:
    key = summary_key(row)
    prompt = build_summary_user_prompt(
        row,
        max_history_items=int(cfg["max_history_items"]),
        review_char_limit=int(cfg["review_char_limit"]),
    )
    max_retries = int(cfg["max_retries"])
    retry_base_seconds = float(cfg["retry_base_seconds"])
    primary_model = cfg["model"]
    fallback_model = cfg.get("fallback_model", "").strip()

    async with sem:
        for attempt in range(max_retries):
            try:
                text, finish_reason = await request_summary_once(client, primary_model, prompt, cfg)
                if text:
                    return key, text, "primary"

                if fallback_model:
                    fb_text, fb_finish_reason = await request_summary_once(client, fallback_model, prompt, cfg)
                    if fb_text:
                        return key, fb_text, "fallback"
                    if attempt == max_retries - 1:
                        print(
                            f"[warn] empty summary for key={key}, "
                            f"primary_finish={finish_reason}, fallback_finish={fb_finish_reason}",
                            flush=True,
                        )
                        return key, DEFAULT_UNAVAILABLE, "unavailable"
                else:
                    if attempt == max_retries - 1:
                        print(f"[warn] empty summary for key={key}, primary_finish={finish_reason}", flush=True)
                        return key, DEFAULT_UNAVAILABLE, "unavailable"
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[error] summarize failed for key={key}: {type(e).__name__}: {e}", flush=True)
                    return key, DEFAULT_UNAVAILABLE, "error"
                await asyncio.sleep(retry_base_seconds * (2 ** attempt))
    return key, DEFAULT_UNAVAILABLE, "unavailable"


async def build_summaries(
    rows: List[dict],
    cache: Dict[str, str],
    cfg: dict,
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
    for row in rows:
        key = summary_key(row)
        cached = cache.get(key)
        if cached is None:
            pending_rows.append(row)
            continue
        if refresh_invalid_cache and is_placeholder_summary(cached):
            pending_rows.append(row)
            refreshed_invalid_count += 1
            continue
        reused_count += 1

    stats = {
        "total_rows": len(rows),
        "pending_rows": len(pending_rows),
        "reused_cache_count": reused_count,
        "refreshed_invalid_cache_count": refreshed_invalid_count,
        "primary_count": 0,
        "fallback_count": 0,
        "unavailable_count": 0,
        "error_count": 0,
    }
    if not pending_rows:
        return cache, stats

    tasks = [asyncio.create_task(summarize_one(client, row, cfg, sem)) for row in pending_rows]
    completed = 0
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Summarizing interests"):
        key, text, source = await fut
        cache[key] = text
        source_key = f"{source}_count"
        if source_key in stats:
            stats[source_key] += 1
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
    interest_block = (
        "Inferred user interests from historical behavior:\n"
        f"{summary_text}\n\n"
    )
    if marker in original_prompt:
        return original_prompt.replace(marker, interest_block + marker)
    return original_prompt + "\n\n" + interest_block


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


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    paths = config["paths"]
    summary_cfg = config["interest_summary"]

    raw_test_path = os.path.join(script_dir, paths["raw_test_source"])
    data_dir = os.path.join(script_dir, paths["data_dir"])
    cache_dir = os.path.join(script_dir, paths["cache_dir"])
    ensure_dir(data_dir)
    ensure_dir(cache_dir)

    raw_rows = load_jsonl(raw_test_path)
    raw_4548_path = os.path.join(data_dir, "test_raw_4548.jsonl")
    dump_jsonl(raw_4548_path, raw_rows)

    cache_path = os.path.join(cache_dir, summary_cfg["cache_file"])
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, "r", encoding="utf-8"))
    else:
        cache = {}

    cache, summary_stats = asyncio.run(build_summaries(raw_rows, cache, summary_cfg, cache_path))

    interest_rows = []
    for row in raw_rows:
        key = summary_key(row)
        summary_text = cache.get(key, DEFAULT_UNAVAILABLE)
        new_row = dict(row)
        new_row["interest_summary"] = summary_text
        new_row["prompt_original"] = row["prompt"]
        new_row["prompt"] = augment_prompt(row["prompt"], summary_text)
        interest_rows.append(new_row)

    interest_4548_path = os.path.join(data_dir, "test_interest_4548.jsonl")
    dump_jsonl(interest_4548_path, interest_rows)

    sasrec_test_path = os.path.join(script_dir, paths["sasrec_test_path"])
    item_mapping_path = os.path.join(script_dir, paths["item_mapping_path"])
    common_raw_rows = align_common_4385(raw_rows, sasrec_test_path, item_mapping_path)
    common_interest_rows = align_common_4385(interest_rows, sasrec_test_path, item_mapping_path)

    raw_common_path = os.path.join(data_dir, "test_raw_common_4385.jsonl")
    interest_common_path = os.path.join(data_dir, "test_interest_common_4385.jsonl")
    dump_jsonl(raw_common_path, common_raw_rows)
    dump_jsonl(interest_common_path, common_interest_rows)

    summary = {
        "raw_test_path": raw_test_path,
        "raw_size_4548": len(raw_rows),
        "raw_common_size": len(common_raw_rows),
        "interest_common_size": len(common_interest_rows),
        "cache_path": cache_path,
        "cache_size": len(cache),
        "raw_4548_path": raw_4548_path,
        "interest_4548_path": interest_4548_path,
        "raw_common_path": raw_common_path,
        "interest_common_path": interest_common_path,
        "summary_generation_stats": summary_stats,
        "summary_models": {
            "primary": summary_cfg["model"],
            "fallback": summary_cfg.get("fallback_model"),
        },
    }
    with open(os.path.join(data_dir, "prepare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
