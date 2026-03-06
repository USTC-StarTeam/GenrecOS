#!/usr/bin/env python3

import json
import os
import re
from typing import Dict, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, rows: List[dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_item_titles(path: str):
    rows = json.load(open(path, "r", encoding="utf-8"))
    item_to_title = {}
    for row in rows:
        item_id = row.get("item_id")
        title = row.get("condensed_title")
        if not item_id or not title:
            continue
        item_to_title[item_id] = title
    title_to_item = {}
    for item_id, title in item_to_title.items():
        title_to_item.setdefault(title, item_id)
    return item_to_title, title_to_item


def normalize_title(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'["“”\'`]', "", text)
    text = re.sub(r"[^a-z0-9\s&+/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_chat_prompt(user_prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful recommendation assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

