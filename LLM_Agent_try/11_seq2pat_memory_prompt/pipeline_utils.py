#!/usr/bin/env python3

import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


SYSTEM_PROMPT = (
    "You are a recommendation model. Given a user's purchase history and optional "
    "behavior-pattern memory, predict the single most likely next product title. "
    "Respond with only the next product title and nothing else."
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
    pattern_matches: List[dict]


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


def format_pattern_text(pattern: dict) -> str:
    titles = pattern.get("pattern_titles") or pattern.get("pattern_item_ids") or []
    chain = " -> ".join(f'"{title}"' for title in titles)
    match_type = pattern.get("match_type", "unknown")
    support = int(pattern.get("support", 0))
    if match_type == "full":
        head = f"full match, support={support}"
    else:
        head = (
            f"partial {int(pattern.get('matched_len', 0))}/{int(pattern.get('pattern_len', 0))}, "
            f"support={support}"
        )
    return f"[{head}] {chain}"


def build_prompt(
    history_titles: List[str],
    history_ratings: List[int],
    history_reviews: List[str],
    pattern_matches: Optional[List[dict]] = None,
    max_patterns_in_prompt: int = 3,
) -> str:
    history_lines = []
    for idx, (title, rating, review) in enumerate(zip(history_titles, history_ratings, history_reviews), 1):
        history_lines.append(f'{idx}. "{title}" (Rating: {rating}) - Review: {review}')

    sections = [
        "User's purchase history:",
        "\n".join(history_lines),
    ]

    selected_patterns = (pattern_matches or [])[:max_patterns_in_prompt]
    if selected_patterns:
        pattern_lines = [f"{idx}. {format_pattern_text(pattern)}" for idx, pattern in enumerate(selected_patterns, 1)]
        sections.extend(
            [
                "",
                "Retrieved behavior transition patterns (global memory mined from training-visible histories):",
                "\n".join(pattern_lines),
                "Use these patterns as auxiliary signals only. Prioritize the user's own history.",
            ]
        )

    sections.extend(
        [
            "",
            "Based on the user's interaction history, predict the next product they would be most interested in purchasing.",
            "",
            "Next product:",
        ]
    )
    return "\n".join(sections)


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


def build_item_integer_mapping(user_sequences: Dict[str, List[Tuple[str, int, float, str]]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    unique_items = sorted({item_id for seq in user_sequences.values() for item_id, _, _, _ in seq})
    item_to_int = {item_id: idx + 1 for idx, item_id in enumerate(unique_items)}
    int_to_item = {idx: item_id for item_id, idx in item_to_int.items()}
    return item_to_int, int_to_item


def build_mining_sequences(
    user_sequences: Dict[str, List[Tuple[str, int, float, str]]],
    item_to_int: Dict[str, int],
    trim_last: int,
    min_len: int,
) -> List[List[int]]:
    mining_sequences = []
    for sequence in user_sequences.values():
        items = [item_id for item_id, _, _, _ in sequence]
        if len(items) <= trim_last:
            continue
        visible = items[:-trim_last] if trim_last > 0 else items
        if len(visible) < min_len:
            continue
        mining_sequences.append([item_to_int[item_id] for item_id in visible if item_id in item_to_int])
    return mining_sequences


def mine_patterns_with_seq2pat(
    mining_sequences: List[List[int]],
    int_to_item: Dict[int, str],
    item_to_title: Dict[str, str],
    min_frequency: int,
    min_pattern_len: int,
    max_pattern_len: int,
    max_patterns: int,
    max_span: int,
    n_jobs: int,
    seed: int,
) -> List[dict]:
    from sequential.seq2pat import Seq2Pat

    miner = Seq2Pat(sequences=mining_sequences, max_span=max_span, n_jobs=n_jobs, seed=seed)
    raw_patterns = miner.get_patterns(min_frequency=min_frequency)

    parsed = []
    for pattern_row in raw_patterns:
        if len(pattern_row) < 2:
            continue
        support = int(pattern_row[-1])
        pattern_ints = [int(x) for x in pattern_row[:-1]]
        pattern_len = len(pattern_ints)
        if pattern_len < min_pattern_len or pattern_len > max_pattern_len:
            continue

        pattern_item_ids = []
        for idx in pattern_ints:
            item_id = int_to_item.get(idx)
            if item_id is None:
                break
            pattern_item_ids.append(item_id)
        if len(pattern_item_ids) != pattern_len:
            continue

        parsed.append(
            {
                "pattern_item_ids": pattern_item_ids,
                "pattern_titles": [item_to_title[item_id] for item_id in pattern_item_ids],
                "support": support,
                "pattern_len": pattern_len,
                "score": round(float(support * pattern_len), 4),
            }
        )

    parsed.sort(
        key=lambda row: (
            -row["support"],
            -row["pattern_len"],
            row["pattern_item_ids"],
        )
    )
    if max_patterns > 0:
        parsed = parsed[:max_patterns]

    for idx, row in enumerate(parsed):
        row["pattern_id"] = idx
    return parsed


def _subsequence_match_len(pattern_item_ids: List[str], history_item_ids: List[str]) -> int:
    p = 0
    for item_id in history_item_ids:
        if p < len(pattern_item_ids) and item_id == pattern_item_ids[p]:
            p += 1
    return p


class PatternMatcher:
    def __init__(
        self,
        patterns: List[dict],
        partial_min_ratio: float,
        partial_min_matched: int,
        max_matches: int,
    ):
        self.partial_min_ratio = partial_min_ratio
        self.partial_min_matched = partial_min_matched
        self.max_matches = max_matches
        self.patterns = patterns
        self.by_first_item: Dict[str, List[dict]] = defaultdict(list)
        for row in patterns:
            if not row["pattern_item_ids"]:
                continue
            self.by_first_item[row["pattern_item_ids"][0]].append(row)

    def match(self, history_item_ids: List[str]) -> List[dict]:
        candidates = {}
        for item_id in history_item_ids:
            for pattern in self.by_first_item.get(item_id, []):
                candidates[pattern["pattern_id"]] = pattern

        matched = []
        for pattern in candidates.values():
            pattern_item_ids = pattern["pattern_item_ids"]
            matched_len = _subsequence_match_len(pattern_item_ids, history_item_ids)
            pattern_len = len(pattern_item_ids)
            if matched_len == pattern_len:
                match_type = "full"
                match_quality = 1.0
            else:
                ratio = matched_len / max(pattern_len, 1)
                if matched_len < self.partial_min_matched or ratio < self.partial_min_ratio:
                    continue
                match_type = "partial"
                match_quality = ratio

            rank_score = match_quality * 1000.0 + math.log1p(pattern["support"]) * 10.0 + pattern_len
            matched.append(
                {
                    "pattern_id": pattern["pattern_id"],
                    "pattern_item_ids": pattern_item_ids,
                    "pattern_titles": pattern["pattern_titles"],
                    "support": pattern["support"],
                    "pattern_len": pattern_len,
                    "matched_len": matched_len,
                    "match_type": match_type,
                    "match_quality": round(float(match_quality), 4),
                    "rank_score": round(float(rank_score), 4),
                }
            )

        matched.sort(
            key=lambda row: (
                -row["rank_score"],
                -row["support"],
                -row["pattern_len"],
                row["pattern_id"],
            )
        )
        return matched[: self.max_matches]


def make_sample(
    split: str,
    user_id: str,
    history: List[Tuple[str, float, str]],
    target_item_id: str,
    item_to_title: Dict[str, str],
    max_review_len: int,
    matcher: PatternMatcher,
    max_patterns_in_prompt: int,
) -> Sample:
    history_item_ids = [item_id for item_id, _, _ in history]
    history_titles = [item_to_title[item_id] for item_id, _, _ in history]
    history_ratings = [int(rating) for _, rating, _ in history]
    history_reviews = [truncate_review(review_text, max_review_len) for _, _, review_text in history]
    pattern_matches = matcher.match(history_item_ids)
    if len(pattern_matches) > max_patterns_in_prompt:
        pattern_matches = pattern_matches[:max_patterns_in_prompt]

    return Sample(
        split=split,
        user_id=user_id,
        history_item_ids=history_item_ids,
        history_titles=history_titles,
        history_ratings=history_ratings,
        history_reviews=history_reviews,
        target_item_id=target_item_id,
        target_title=item_to_title[target_item_id],
        pattern_matches=pattern_matches,
    )


def build_split_samples(
    user_sequences: Dict[str, List[Tuple[str, int, float, str]]],
    item_to_title: Dict[str, str],
    max_history: int,
    max_review_len: int,
    matcher: PatternMatcher,
    max_patterns_in_prompt: int,
) -> List[Sample]:
    samples: List[Sample] = []
    for user_id, sequence in user_sequences.items():
        if len(sequence) < 3:
            continue
        items = [(item_id, rating, review_text) for item_id, _, rating, review_text in sequence]

        samples.append(
            make_sample(
                split="test",
                user_id=user_id,
                history=items[:-1][-max_history:],
                target_item_id=items[-1][0],
                item_to_title=item_to_title,
                max_review_len=max_review_len,
                matcher=matcher,
                max_patterns_in_prompt=max_patterns_in_prompt,
            )
        )
        samples.append(
            make_sample(
                split="val",
                user_id=user_id,
                history=items[:-2][-max_history:],
                target_item_id=items[-2][0],
                item_to_title=item_to_title,
                max_review_len=max_review_len,
                matcher=matcher,
                max_patterns_in_prompt=max_patterns_in_prompt,
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
                    max_review_len=max_review_len,
                    matcher=matcher,
                    max_patterns_in_prompt=max_patterns_in_prompt,
                )
            )
    return samples


def sample_to_row(sample: Sample, max_patterns_in_prompt: int) -> dict:
    prompt = build_prompt(
        sample.history_titles,
        sample.history_ratings,
        sample.history_reviews,
        pattern_matches=sample.pattern_matches,
        max_patterns_in_prompt=max_patterns_in_prompt,
    )
    return {
        "split": sample.split,
        "user_id": sample.user_id,
        "history_item_ids": sample.history_item_ids,
        "history_titles": sample.history_titles,
        "history_ratings": sample.history_ratings,
        "history_reviews": sample.history_reviews,
        "target_item_id": sample.target_item_id,
        "target_title": sample.target_title,
        "pattern_matches": sample.pattern_matches,
        "num_pattern_matches": len(sample.pattern_matches),
        "prompt": prompt,
    }


def summarize_rows(rows: List[dict]) -> dict:
    history_lengths = [len(row["history_titles"]) for row in rows]
    prompt_chars = [len(row["prompt"]) for row in rows]
    pattern_match_counts = [int(row.get("num_pattern_matches", 0)) for row in rows]
    full_match_samples = 0
    partial_match_samples = 0
    for row in rows:
        types = {m.get("match_type") for m in row.get("pattern_matches", [])}
        if "full" in types:
            full_match_samples += 1
        if "partial" in types:
            partial_match_samples += 1

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
        "pattern_matches_avg": round(sum(pattern_match_counts) / len(pattern_match_counts), 3)
        if pattern_match_counts
        else 0.0,
        "pattern_matches_nonzero_ratio": round(
            sum(1 for x in pattern_match_counts if x > 0) / len(pattern_match_counts), 4
        )
        if pattern_match_counts
        else 0.0,
        "full_match_sample_ratio": round(full_match_samples / len(rows), 4) if rows else 0.0,
        "partial_match_sample_ratio": round(partial_match_samples / len(rows), 4) if rows else 0.0,
    }
