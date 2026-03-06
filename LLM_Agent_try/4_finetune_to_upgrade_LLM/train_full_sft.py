#!/usr/bin/env python3

import argparse
import json
import math
import os
from typing import Dict, List, Optional

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from pipeline_utils import ensure_dir, format_chat_example, format_chat_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Full-parameter SFT for title-only next-item prediction.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sft_config.yaml"),
        help="Path to config yaml.",
    )
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
            seq_len = len(feature["input_ids"])
            pad_len = max_length - seq_len
            input_ids.append(feature["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def choose_optimizer(training_cfg: dict) -> str:
    explicit = training_cfg.get("optimizer", "auto")
    if explicit != "auto":
        return explicit
    if not torch.cuda.is_available():
        return "adamw_torch"
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return "adamw_torch_fused" if total_gb >= 90 else "adafactor"


def choose_attention_impl(training_cfg: dict) -> Optional[str]:
    explicit = training_cfg.get("attn_implementation", "auto")
    if explicit != "auto":
        return explicit
    if torch.cuda.is_available() and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"
    return None


def preprocess_example(example, tokenizer, max_length: int):
    prompt_text = format_chat_prompt(example["prompt"])
    full_text = format_chat_example(example["prompt"], example["target_title"])

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    original_full_len = len(full_ids)
    truncate_from_left = max(0, original_full_len - max_length)
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


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths_cfg = config["paths"]
    train_cfg = config["training"]

    data_dir = os.path.join(script_dir, paths_cfg["data_dir"])
    cache_dir = os.path.join(script_dir, paths_cfg["cache_dir"])
    output_dir = os.path.join(script_dir, paths_cfg["output_dir"])
    logging_dir = os.path.join(script_dir, paths_cfg["logging_dir"])
    ensure_dir(cache_dir)
    ensure_dir(output_dir)
    ensure_dir(logging_dir)

    if train_cfg.get("tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model_path = os.path.join(script_dir, paths_cfg["base_model_path"])
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if train_cfg.get("bf16", True) else torch.float16
    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    attn_implementation = choose_attention_impl(train_cfg)
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.config.use_cache = False

    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    dataset = load_dataset(
        "json",
        data_files={"train": train_path, "validation": val_path},
        cache_dir=cache_dir,
    )

    max_length = train_cfg["max_length"]
    num_proc = train_cfg.get("preprocess_num_proc", 1)

    # Older `datasets` is more stable with explicit python-side preprocessing.
    def preprocess_batch(batch):
        examples = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        processed = [preprocess_example(example, tokenizer, max_length) for example in examples]
        return {key: [row[key] for row in processed] for key in processed[0]}

    tokenized = dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=128,
        num_proc=num_proc,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing SFT dataset",
        load_from_cache_file=True,
    )

    total_train_samples = len(tokenized["train"])
    per_device_bs = train_cfg["per_device_train_batch_size"]
    grad_accum = train_cfg["gradient_accumulation_steps"]
    world_size = 1
    steps_per_epoch = math.ceil(total_train_samples / (per_device_bs * grad_accum * world_size))
    eval_steps = train_cfg.get("eval_steps") or steps_per_epoch
    save_steps = train_cfg.get("save_steps") or eval_steps

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        logging_steps=int(train_cfg["logging_steps"]),
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=int(eval_steps),
        save_steps=int(save_steps),
        save_total_limit=int(train_cfg["save_total_limit"]),
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 2)),
        dataloader_pin_memory=True,
        dataloader_persistent_workers=bool(train_cfg.get("dataloader_num_workers", 2) > 0),
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim=choose_optimizer(train_cfg),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        report_to=[],
        logging_dir=logging_dir,
        seed=int(train_cfg["seed"]),
        data_seed=int(train_cfg["seed"]),
        save_safetensors=True,
        group_by_length=True,
        auto_find_batch_size=bool(train_cfg.get("auto_find_batch_size", False)),
    )

    collator = CompletionOnlyCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=int(train_cfg.get("early_stopping_patience", 3))),
            GpuStatsCallback(),
        ],
    )

    train_result = trainer.train(resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"))
    trainer.save_state()
    eval_metrics = trainer.evaluate()

    best_model_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    metrics_path = os.path.join(output_dir, "train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_result": train_result.metrics,
                "eval_metrics": eval_metrics,
                "optimizer": training_args.optim,
                "attn_implementation": attn_implementation or "default",
                "steps_per_epoch": steps_per_epoch,
            },
            f,
            indent=2,
        )

    print(f"Best model saved to {best_model_dir}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
