import json
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, HfArgumentParser
import pandas as pd
from datasets import Dataset
from peft import get_peft_model, LoraConfig
import os
import torch
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PathArguments:
    rec_model_path: str
    train_data_path: str
    val_data_path: str
    special_tokens_path: str

@dataclass
class LoraArguments:
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]

def prepare_chat_dataset(data_path):
    data_pq = pd.read_json(data_path, lines=True)
    texts = []
    system_message = "You are a professional recommendation expert who needs to recommend the next possible purchase for users based on their purchase history. Please predict the most likely next product that the user will purchase based on the user's historical purchase information."
    for _, row in data_pq.iterrows():
        if('title' in row.keys() and row['title'] is not None):
            title = row['title'][0]
            categories = row['categories'][0]
            assistant_content = f"<think>\nThe user is likely to buy items in {categories} category\n</think>\n{row['groundtruth'][0]}"
        else:
            assistant_content = f"<think>\n\n</think>\n{row['groundtruth'][0]}"
        formatted_text = (
            f"<|im_start|>system\n{system_message}<|im_end|>\n"
            f"<|im_start|>user\n{row['description']}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_content}<|im_end|>\n"
        )
        texts.append(formatted_text)
    dataset_dict = {
        'text': texts
    }
    return Dataset.from_dict(dataset_dict)

def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples['text'],
        padding='longest',
        truncation=True,
        max_length=4096,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    return tokenized

class CustomDataCollator:
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]

        max_length = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_attention_mask = []
        labels = []

        for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
            padding_length = max_length - len(ids)
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            padded_mask = mask + [0] * padding_length

            label = padded_ids.copy()

            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            user_start_pos = text.find("<|im_start|>user")

            if user_start_pos != -1:
                user_start_tokens = self.tokenizer.encode("<|im_start|>user", add_special_tokens=False)

                for j in range(len(ids) - len(user_start_tokens) + 1):
                    if ids[j:j+len(user_start_tokens)] == user_start_tokens:
                        for k in range(j):
                            label[k] = -100
                        break
                else:
                    for k in range(len(label)):
                        label[k] = -100
            else:
                for k in range(len(label)):
                    label[k] = -100

            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
            labels.append(label)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


if __name__ == "__main__":
    parser = HfArgumentParser((PathArguments, LoraArguments, TrainingArguments))
    yaml_file = None
    for arg in sys.argv:
        if arg.endswith((".yaml", ".yml")):
            yaml_file = arg
            break

    if yaml_file is not None:
        path_args, lora_args, training_args = parser.parse_yaml_file(yaml_file=yaml_file)
    else:
        path_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    rec_model_path = Path(path_args.rec_model_path).resolve()
    train_data_path = Path(path_args.train_data_path).resolve()
    val_data_path = Path(path_args.val_data_path).resolve()
    special_tokens_path = Path(path_args.special_tokens_path).resolve()

    rec_model = AutoModelForCausalLM.from_pretrained(str(rec_model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(rec_model_path))
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    with open(special_tokens_path, "r") as f:
        special_tokens = json.load(f)
    tokenized_special_tokens = tokenizer.convert_tokens_to_ids(special_tokens)
    # 进行cot训练
    training_args.label_names = ["labels"]

    lora_congig = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        task_type="CAUSAL_LM",
        bias="none",
        trainable_token_indices={
            "embed_tokens": tokenized_special_tokens
        }
    )
    model = get_peft_model(rec_model, lora_congig)
    

    train_dataset = prepare_chat_dataset(train_data_path)
    val_dataset = prepare_chat_dataset(val_data_path)

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data"
    )

    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation data"
    )

    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    result = trainer.evaluate()
    print(result)

    #    
    # model = model.merge_and_unload()
    # output_dir = os.path.join(training_args.output_dir, "best_model")
    # tokenizer.save_pretrained(output_dir)
    # model.save_pretrained(output_dir)

    # 1. 显式等待所有进程到达此处，防止有的进程还在 eval 没跑完
    trainer.accelerator.wait_for_everyone()

    # 注意：merge_and_unload 会在每个进程的显存/内存中进行合并。
    # 如果显存非常紧张，这里可能会 OOM。如果是 DeepSpeed Zero3，这里写法会更复杂。
    # 既然你前面跑通了，说明显存够用。
    model = model.merge_and_unload()

    # 2. 只有主进程负责保存模型和 Tokenizer，避免文件写入冲突
    output_dir = os.path.join(training_args.output_dir, "best_model")
    
    if trainer.accelerator.is_main_process:
        print(f"Saving merged model to {output_dir} ...")
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        print("Model saved successfully.")

    # 3. 再次等待。确保 Rank 0 保存完文件之前，其他进程不要退出。
    # 这一步是为了防止 Rank 0 还在写文件，其他进程退出导致通信组崩溃（虽然在这里主要是 Rank 0 退出导致别人崩溃）。
    # 主要是为了优雅退出。
    trainer.accelerator.wait_for_everyone()
    
    

    
