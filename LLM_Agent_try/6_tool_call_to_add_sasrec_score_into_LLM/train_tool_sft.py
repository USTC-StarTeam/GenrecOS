#!/usr/bin/env python3

import argparse
import json
import math
import os
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from common import (
    TOOL_TOKEN,
    choose_model_path,
    ensure_dir,
    get_track_best_model_dir,
    get_track_dir,
    get_track_output_dir,
    load_causal_model,
    set_global_seed,
    tool_chat_example,
    tool_chat_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tool-aware SFT model.")
    parser.add_argument("--track_name", type=str, choices=["pre_sft", "post_sft"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=1232)
    parser.add_argument("--preprocess_num_proc", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--learning_rate", type=float, default=1.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=6)
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    return parser.parse_args()


class GpuStatsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not torch.cuda.is_available():
            return
        logs = logs or {}
        logs["gpu_mem_alloc_gb"] = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
        logs["gpu_mem_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024 ** 3), 2)


class CompletionOnlyCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of and max_length % self.pad_to_multiple_of != 0:
            max_length = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        input_ids, attention_mask, labels = [], [], []
        pad_token_id = self.tokenizer.pad_token_id
        for feature in features:
            pad_len = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def preprocess_example(example, tokenizer, max_length: int):
    prompt_text = tool_chat_prompt(example["prompt"])
    full_text = tool_chat_example(example["prompt"], example["assistant_target"])

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    truncate_from_left = max(0, len(full_ids) - max_length)
    if truncate_from_left > 0:
        full_ids = full_ids[-max_length:]
    prompt_prefix = max(0, len(prompt_ids) - truncate_from_left)

    labels = [-100] * min(prompt_prefix, len(full_ids)) + full_ids[min(prompt_prefix, len(full_ids)):]
    attention_mask = [1] * len(full_ids)
    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "length": len(full_ids),
    }


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    data_dir = get_track_dir(args.track_name)
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")
    output_dir = get_track_output_dir(args.track_name)
    cache_dir = os.path.join(output_dir, "cache")
    logging_dir = os.path.join(output_dir, "logs")
    ensure_dir(cache_dir)
    ensure_dir(output_dir)
    ensure_dir(logging_dir)

    model_path = choose_model_path(args.track_name)
    model, tokenizer = load_causal_model(
        model_path=model_path,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        add_tool_token=True,
        train_mode=True,
        gradient_checkpointing=True,
    )
    tokenizer.padding_side = "right"
    model.train()

    dataset = load_dataset(
        "json",
        data_files={"train": train_path, "validation": val_path},
        cache_dir=cache_dir,
    )

    def preprocess_batch(batch):
        examples = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        processed = [preprocess_example(example, tokenizer, args.max_length) for example in examples]
        return {key: [row[key] for row in processed] for key in processed[0]}

    tokenized = dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=128,
        num_proc=args.preprocess_num_proc,
        remove_columns=dataset["train"].column_names,
        desc=f"Tokenizing {args.track_name}",
        load_from_cache_file=True,
    )

    total_train_samples = len(tokenized["train"])
    steps_per_epoch = math.ceil(
        total_train_samples / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    eval_steps = args.eval_steps or steps_per_epoch
    save_steps = args.save_steps or eval_steps

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=float(args.learning_rate),
        num_train_epochs=float(args.num_train_epochs),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        logging_steps=int(args.logging_steps),
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=int(eval_steps),
        save_steps=int(save_steps),
        save_total_limit=int(args.save_total_limit),
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=int(args.dataloader_num_workers),
        dataloader_pin_memory=True,
        dataloader_persistent_workers=bool(args.dataloader_num_workers > 0),
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        lr_scheduler_type="cosine",
        report_to=[],
        logging_dir=logging_dir,
        seed=int(args.seed),
        data_seed=int(args.seed),
        save_safetensors=True,
        group_by_length=True,
        auto_find_batch_size=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=CompletionOnlyCollator(tokenizer=tokenizer, pad_to_multiple_of=8),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=int(args.early_stopping_patience)),
            GpuStatsCallback(),
        ],
    )

    train_result = trainer.train()
    trainer.save_state()
    eval_metrics = trainer.evaluate()

    best_model_dir = get_track_best_model_dir(args.track_name)
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    metrics = {
        "track_name": args.track_name,
        "base_checkpoint": model_path,
        "tool_token": TOOL_TOKEN,
        "train_result": train_result.metrics,
        "eval_metrics": eval_metrics,
        "best_model_dir": best_model_dir,
    }
    with open(os.path.join(output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

