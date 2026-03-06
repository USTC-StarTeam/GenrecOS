#!/usr/bin/env python3

import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


SYSTEM_PROMPT = (
    "You are a recommendation model. Given a user's purchase history, predict the "
    "single most likely next product title. Respond with only the next product "
    "title and nothing else."
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def dump_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)


def normalize_title(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    return text


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


def truncate_review(review_text: str, max_review_len: int) -> str:
    if len(review_text) <= max_review_len:
        return review_text
    return review_text[: max_review_len - 3] + "..."


def build_prompt(history_titles: List[str], history_ratings: List[int], history_reviews: List[str]) -> str:
    history_lines = []
    for idx, (title, rating, review) in enumerate(zip(history_titles, history_ratings, history_reviews), 1):
        history_lines.append(f'{idx}. "{title}" (Rating: {rating}) - Review: {review}')
    history_block = "\n".join(history_lines)
    return (
        "User's purchase history:\n"
        f"{history_block}\n\n"
        "Based on the user's interaction history, predict the next product they would be most interested in purchasing.\n\n"
        "Next product:"
    )


def format_chat_prompt(prompt: str) -> str:
    return (
        f"{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n"
        f"{IM_START}user\n{prompt}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def format_chat_example(prompt: str, target: str) -> str:
    return f"{format_chat_prompt(prompt)}{target}{IM_END}\n"


def parse_raw_sequences(
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


def build_split_samples(
    user_sequences: Dict[str, List[Tuple[str, int, float, str]]],
    item_to_title: Dict[str, str],
    max_history: int,
    max_review_len: int,
) -> List[Sample]:
    samples: List[Sample] = []

    for user_id, sequence in user_sequences.items():
        if len(sequence) < 3:
            continue

        items = [(item_id, rating, review_text) for item_id, _, rating, review_text in sequence]

        test_history = items[:-1][-max_history:]
        test_target_item = items[-1][0]
        samples.append(
            make_sample(
                split="test",
                user_id=user_id,
                history=test_history,
                target_item_id=test_target_item,
                item_to_title=item_to_title,
                max_review_len=max_review_len,
            )
        )

        val_history = items[:-2][-max_history:]
        val_target_item = items[-2][0]
        samples.append(
            make_sample(
                split="val",
                user_id=user_id,
                history=val_history,
                target_item_id=val_target_item,
                item_to_title=item_to_title,
                max_review_len=max_review_len,
            )
        )

        train_items = items[:-2]
        for target_idx in range(1, len(train_items)):
            history = train_items[:target_idx][-max_history:]
            target_item_id = train_items[target_idx][0]
            samples.append(
                make_sample(
                    split="train",
                    user_id=user_id,
                    history=history,
                    target_item_id=target_item_id,
                    item_to_title=item_to_title,
                    max_review_len=max_review_len,
                )
            )

    return samples


def make_sample(
    split: str,
    user_id: str,
    history: List[Tuple[str, float, str]],
    target_item_id: str,
    item_to_title: Dict[str, str],
    max_review_len: int,
) -> Sample:
    history_item_ids = [item_id for item_id, _, _ in history]
    history_titles = [item_to_title[item_id] for item_id, _, _ in history]
    history_ratings = [int(rating) for _, rating, _ in history]
    history_reviews = [truncate_review(review_text, max_review_len) for _, _, review_text in history]
    return Sample(
        split=split,
        user_id=user_id,
        history_item_ids=history_item_ids,
        history_titles=history_titles,
        history_ratings=history_ratings,
        history_reviews=history_reviews,
        target_item_id=target_item_id,
        target_title=item_to_title[target_item_id],
    )


def sample_to_row(sample: Sample) -> dict:
    prompt = build_prompt(sample.history_titles, sample.history_ratings, sample.history_reviews)
    return {
        "split": sample.split,
        "user_id": sample.user_id,
        "history_item_ids": sample.history_item_ids,
        "history_titles": sample.history_titles,
        "history_ratings": sample.history_ratings,
        "history_reviews": sample.history_reviews,
        "target_item_id": sample.target_item_id,
        "target_title": sample.target_title,
        "prompt": prompt,
    }


def summarize_rows(rows: List[dict]) -> dict:
    history_lengths = [len(row["history_titles"]) for row in rows]
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
        "prompt_chars_avg": round(sum(prompt_chars) / len(prompt_chars), 2) if prompt_chars else 0.0,
    }
