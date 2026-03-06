#!/usr/bin/env python3

import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

SYSTEM_PROMPT = (
    "You are a recommendation model. Given a user's purchase history, predict the "
    "single most likely next product title. Respond with only the next product "
    "title and nothing else."
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


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


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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


def format_chat_prompt(prompt: str) -> str:
    return (
        f"{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n"
        f"{IM_START}user\n{prompt}{IM_END}\n"
        f"{IM_START}assistant\n"
    )
