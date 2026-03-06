#!/usr/bin/env python3

import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


SYSTEM_PROMPT = (
    "You are a recommendation model. Given a user's purchase history and optional "
    "supporting reviews from other users on those products, predict the single "
    "most likely next product title. Respond with only the next product title and nothing else."
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


@dataclass
class Sample:
    split: str
    user_id: str
    history_item_ids: List[str]
    history_titles: List[str]
    history_ratings: List[int]
    history_reviews: List[str]
    target_item_id: str
    target_title: str
    retrieved_reviews: List[dict]
    prompt: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, data) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def dump_jsonl(path: str, rows: Iterable[dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)


def normalize_title(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_review(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_review(review_text: str, max_review_len: int) -> str:
    review_text = normalize_review(review_text)
    if len(review_text) <= max_review_len:
        return review_text
    return review_text[: max_review_len - 3] + "..."


def load_item_titles(titles_path: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    titles_data = load_json(titles_path)
    item_to_title = {}
    norm_title_to_items = defaultdict(list)
    for row in titles_data:
        item_id = row["item_id"]
        title = normalize_title(row["condensed_title"])
        item_to_title[item_id] = title
        norm_title_to_items[title.lower()].append(item_id)
    return item_to_title, dict(norm_title_to_items)


def review_quality(review_text: str) -> float:
    length = len(review_text)
    capped = min(length, 180)
    closeness = abs(length - 110)
    return capped - 0.25 * closeness


def rating_bucket(rating: int) -> int:
    if rating <= 2:
        return -1
    if rating == 3:
        return 0
    return 1


def parse_filtered_sequences(
    raw_data_path: str,
    item_to_title: Dict[str, str],
    min_item_freq: int,
    min_seq_length: int,
) -> Dict[str, List[Tuple[str, int, float, str]]]:
    item_freq = Counter()
    user_sequences = defaultdict(list)

    with open(raw_data_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            user_id = row.get("user_id", "")
            item_id = row.get("parent_asin") or row.get("asin", "")
            timestamp = row.get("timestamp", 0)
            rating = row.get("rating", 0)
            review_text = row.get("text", "")

            if not user_id or not item_id:
                continue
            if item_id not in item_to_title:
                continue

            user_sequences[user_id].append((item_id, int(timestamp), float(rating), review_text))
            item_freq[item_id] += 1

    valid_items = {item_id for item_id, freq in item_freq.items() if freq >= min_item_freq}
    filtered_sequences = {}
    for user_id, sequence in user_sequences.items():
        filtered = [row for row in sequence if row[0] in valid_items]
        if len(filtered) >= min_seq_length:
            filtered.sort(key=lambda x: x[1])
            filtered_sequences[user_id] = filtered
    return filtered_sequences


def build_item_review_pool(
    user_sequences: Dict[str, List[Tuple[str, int, float, str]]],
    item_to_title: Dict[str, str],
    retrieved_review_max_len: int,
    pool_size_per_item: int,
    min_review_chars: int = 0,
    min_review_quality: float = 0.0,
) -> Dict[str, List[dict]]:
    item_user_best = defaultdict(dict)
    for user_id, sequence in user_sequences.items():
        for item_id, _, rating, review_text in sequence:
            review_text = truncate_review(review_text, retrieved_review_max_len)
            if not review_text:
                continue
            quality = review_quality(review_text)
            if len(review_text) < min_review_chars:
                continue
            if quality < min_review_quality:
                continue
            candidate = {
                "user_id": user_id,
                "item_id": item_id,
                "item_title": item_to_title[item_id],
                "rating": int(rating),
                "review_text": review_text,
                "quality": quality,
            }
            prev = item_user_best[item_id].get(user_id)
            if prev is None or candidate["quality"] > prev["quality"]:
                item_user_best[item_id][user_id] = candidate

    item_review_pool = {}
    for item_id, by_user in item_user_best.items():
        sorted_candidates = sorted(
            by_user.values(),
            key=lambda row: (-row["quality"], row["user_id"]),
        )
        item_review_pool[item_id] = sorted_candidates[:pool_size_per_item]
    return item_review_pool


def select_retrieved_reviews(
    user_id: str,
    history_item_ids: List[str],
    history_ratings: List[int],
    item_to_title: Dict[str, str],
    item_review_pool: Dict[str, List[dict]],
    short_history_threshold: int,
    max_aug_reviews_per_sample: int,
    max_aug_reviews_per_item: int,
    recent_first: bool,
    recent_item_window: Optional[int] = None,
    require_same_rating_bucket: bool = False,
) -> List[dict]:
    if len(history_item_ids) > short_history_threshold:
        return []

    ordered_pairs = list(zip(history_item_ids, history_ratings))
    if recent_first:
        ordered_pairs = list(reversed(ordered_pairs))
    if recent_item_window is not None and recent_item_window > 0:
        ordered_pairs = ordered_pairs[:recent_item_window]

    retrieved = []
    used_reviews = set()
    for item_id, user_rating in ordered_pairs:
        per_item = 0
        for candidate in item_review_pool.get(item_id, []):
            if candidate["user_id"] == user_id:
                continue
            if require_same_rating_bucket and rating_bucket(candidate["rating"]) != rating_bucket(user_rating):
                continue
            review_key = (candidate["user_id"], candidate["review_text"])
            if review_key in used_reviews:
                continue
            retrieved.append(
                {
                    "source_item_id": item_id,
                    "source_item_title": item_to_title[item_id],
                    "review_user_id": candidate["user_id"],
                    "review_rating": candidate["rating"],
                    "review_text": candidate["review_text"],
                }
            )
            used_reviews.add(review_key)
            per_item += 1
            if per_item >= max_aug_reviews_per_item:
                break
            if len(retrieved) >= max_aug_reviews_per_sample:
                return retrieved
        if len(retrieved) >= max_aug_reviews_per_sample:
            break
    return retrieved


def build_augmented_prompt(
    history_titles: List[str],
    history_ratings: List[int],
    history_reviews: List[str],
    retrieved_reviews: List[dict],
) -> str:
    history_lines = []
    for idx, (title, rating, review) in enumerate(zip(history_titles, history_ratings, history_reviews), 1):
        history_lines.append(f'{idx}. "{title}" (Rating: {rating}) - Review: {review}')

    prompt_parts = [
        "User's purchase history:",
        "\n".join(history_lines),
    ]

    if retrieved_reviews:
        retrieval_lines = []
        for idx, row in enumerate(retrieved_reviews, 1):
            retrieval_lines.append(
                f'{idx}. For "{row["source_item_title"]}", another user rated it {row["review_rating"]} and said: {row["review_text"]}'
            )
        prompt_parts.extend(
            [
                "",
                "Additional reviews from other users on products this user has already interacted with:",
                "\n".join(retrieval_lines),
            ]
        )

    prompt_parts.extend(
        [
            "",
            "Based on the user's interaction history, predict the next product they would be most interested in purchasing.",
            "",
            "Next product:",
        ]
    )
    return "\n".join(prompt_parts)


def format_chat_prompt(prompt: str) -> str:
    return (
        f"{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n"
        f"{IM_START}user\n{prompt}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def format_chat_example(prompt: str, target: str) -> str:
    return f"{format_chat_prompt(prompt)}{target}{IM_END}\n"


def build_split_samples(
    user_sequences: Dict[str, List[Tuple[str, int, float, str]]],
    item_to_title: Dict[str, str],
    item_review_pool: Dict[str, List[dict]],
    max_history: int,
    max_review_len: int,
    short_history_threshold: int,
    max_aug_reviews_per_sample: int,
    max_aug_reviews_per_item: int,
    recent_first: bool,
    recent_item_window: Optional[int] = None,
    require_same_rating_bucket: bool = False,
) -> List[Sample]:
    samples: List[Sample] = []

    for user_id, sequence in user_sequences.items():
        items = [(item_id, rating, review_text) for item_id, _, rating, review_text in sequence]
        if len(items) < 3:
            continue

        samples.append(
            make_sample(
                split="test",
                user_id=user_id,
                history=items[:-1][-max_history:],
                target_item_id=items[-1][0],
                item_to_title=item_to_title,
                item_review_pool=item_review_pool,
                max_review_len=max_review_len,
                short_history_threshold=short_history_threshold,
                max_aug_reviews_per_sample=max_aug_reviews_per_sample,
                max_aug_reviews_per_item=max_aug_reviews_per_item,
                recent_first=recent_first,
                recent_item_window=recent_item_window,
                require_same_rating_bucket=require_same_rating_bucket,
            )
        )
        samples.append(
            make_sample(
                split="val",
                user_id=user_id,
                history=items[:-2][-max_history:],
                target_item_id=items[-2][0],
                item_to_title=item_to_title,
                item_review_pool=item_review_pool,
                max_review_len=max_review_len,
                short_history_threshold=short_history_threshold,
                max_aug_reviews_per_sample=max_aug_reviews_per_sample,
                max_aug_reviews_per_item=max_aug_reviews_per_item,
                recent_first=recent_first,
                recent_item_window=recent_item_window,
                require_same_rating_bucket=require_same_rating_bucket,
            )
        )

        train_items = items[:-2]
        for target_idx in range(1, len(train_items)):
            samples.append(
                make_sample(
                    split="train",
                    user_id=user_id,
                    history=train_items[:target_idx][-max_history:],
                    target_item_id=train_items[target_idx][0],
                    item_to_title=item_to_title,
                    item_review_pool=item_review_pool,
                    max_review_len=max_review_len,
                    short_history_threshold=short_history_threshold,
                    max_aug_reviews_per_sample=max_aug_reviews_per_sample,
                    max_aug_reviews_per_item=max_aug_reviews_per_item,
                    recent_first=recent_first,
                    recent_item_window=recent_item_window,
                    require_same_rating_bucket=require_same_rating_bucket,
                )
            )
    return samples


def make_sample(
    split: str,
    user_id: str,
    history: List[Tuple[str, float, str]],
    target_item_id: str,
    item_to_title: Dict[str, str],
    item_review_pool: Dict[str, List[dict]],
    max_review_len: int,
    short_history_threshold: int,
    max_aug_reviews_per_sample: int,
    max_aug_reviews_per_item: int,
    recent_first: bool,
    recent_item_window: Optional[int] = None,
    require_same_rating_bucket: bool = False,
) -> Sample:
    history_item_ids = [item_id for item_id, _, _ in history]
    history_titles = [item_to_title[item_id] for item_id, _, _ in history]
    history_ratings = [int(rating) for _, rating, _ in history]
    history_reviews = [truncate_review(review_text, max_review_len) for _, _, review_text in history]
    retrieved_reviews = select_retrieved_reviews(
        user_id=user_id,
        history_item_ids=history_item_ids,
        history_ratings=history_ratings,
        item_to_title=item_to_title,
        item_review_pool=item_review_pool,
        short_history_threshold=short_history_threshold,
        max_aug_reviews_per_sample=max_aug_reviews_per_sample,
        max_aug_reviews_per_item=max_aug_reviews_per_item,
        recent_first=recent_first,
        recent_item_window=recent_item_window,
        require_same_rating_bucket=require_same_rating_bucket,
    )
    prompt = build_augmented_prompt(history_titles, history_ratings, history_reviews, retrieved_reviews)
    return Sample(
        split=split,
        user_id=user_id,
        history_item_ids=history_item_ids,
        history_titles=history_titles,
        history_ratings=history_ratings,
        history_reviews=history_reviews,
        target_item_id=target_item_id,
        target_title=item_to_title[target_item_id],
        retrieved_reviews=retrieved_reviews,
        prompt=prompt,
    )


def sample_to_row(sample: Sample) -> dict:
    return {
        "split": sample.split,
        "user_id": sample.user_id,
        "history_item_ids": sample.history_item_ids,
        "history_titles": sample.history_titles,
        "history_ratings": sample.history_ratings,
        "history_reviews": sample.history_reviews,
        "target_item_id": sample.target_item_id,
        "target_title": sample.target_title,
        "retrieved_reviews": sample.retrieved_reviews,
        "num_retrieved_reviews": len(sample.retrieved_reviews),
        "prompt": sample.prompt,
    }


def summarize_rows(rows: List[dict]) -> dict:
    history_lengths = [len(row["history_titles"]) for row in rows]
    retrieved_counts = [row["num_retrieved_reviews"] for row in rows]
    prompt_chars = [len(row["prompt"]) for row in rows]
    unique_users = len({row["user_id"] for row in rows})
    unique_targets = len({row["target_item_id"] for row in rows})
    return {
        "samples": len(rows),
        "unique_users": unique_users,
        "unique_targets": unique_targets,
        "history_len_min": min(history_lengths) if history_lengths else 0,
        "history_len_avg": round(sum(history_lengths) / len(history_lengths), 2) if history_lengths else 0.0,
        "history_len_max": max(history_lengths) if history_lengths else 0,
        "retrieved_reviews_avg": round(sum(retrieved_counts) / len(retrieved_counts), 3) if retrieved_counts else 0.0,
        "retrieved_reviews_positive_rate": round(
            sum(1 for x in retrieved_counts if x > 0) / len(retrieved_counts), 4
        ) if retrieved_counts else 0.0,
        "prompt_chars_avg": round(sum(prompt_chars) / len(prompt_chars), 2) if prompt_chars else 0.0,
    }
