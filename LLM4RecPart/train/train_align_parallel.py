import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from datasets import Dataset
from peft import TrainableTokensConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

@dataclass
class PathArguments:
    expanded_model_path: str
    train_data_path: str
    val_data_path: str
    special_tokens_path: str

def prepare_dataset(data_path, local_rank=0):
    if local_rank == 0:
        print(f"Loading data from: {data_path}")
    
    df_j = pd.read_json(data_path, lines=True)
    
    if local_rank == 0:
        print(f"Data shape: {df_j.shape}")
        print(f"Columns: {list(df_j.columns)}")
    
    texts = df_j['description']
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

def main():
    parser = HfArgumentParser((PathArguments, TrainingArguments))
    yaml_file = None
    for arg in sys.argv:
        if arg.endswith((".yaml", ".yml")):
            yaml_file = arg
            break

    if yaml_file is not None:
        path_args, training_args = parser.parse_yaml_file(yaml_file=yaml_file)
    else:
        path_args, training_args = parser.parse_args_into_dataclasses()

    # 获取当前进程的 rank，-1 表示非分布式，0 表示主进程
    local_rank = training_args.local_rank
    
    # 设置 label_names 确保 loss 计算正确
    training_args.label_names = ["labels"]

    # 路径解析
    expanded_model_path = Path(path_args.expanded_model_path).resolve()
    train_data_path = Path(path_args.train_data_path).resolve()
    val_data_path = Path(path_args.val_data_path).resolve()
    special_tokens_path = Path(path_args.special_tokens_path).resolve()

    # --- 1. 仅主进程打印调试信息 ---
    if local_rank in [-1, 0]:
        print(f"Using model_dir: {expanded_model_path}")
        print(f"Training data path: {train_data_path}")
        print(f"Validation data path: {val_data_path}")
        print(f"Special tokens path: {special_tokens_path}")
        print(f"Output directory: {training_args.output_dir}")

    # --- 2. 加载模型与分词器 ---
    # AutoModel 会根据环境变量自动处理 device_map，DDP模式下通常不需要手动指定 device_map="auto"
    expanded_model = AutoModelForCausalLM.from_pretrained(
        expanded_model_path,
        torch_dtype="auto" # 建议加上 dtype 自适应
    )
    tokenizer = AutoTokenizer.from_pretrained(expanded_model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. 处理特殊 Token (加入 Script 1 的安全性检查) ---
    with open(special_tokens_path, "r") as f:
        special_tokens = json.load(f)
        
    tokenized_special_tokens_raw = tokenizer.convert_tokens_to_ids(special_tokens)
    
    # 过滤掉 unk_token，防止 Tokenizer 没扩充好导致训练错误
    valid_special_token_ids = []
    skipped_tokens = []
    
    for i, token_id in enumerate(tokenized_special_tokens_raw):
        if token_id != tokenizer.unk_token_id:
            valid_special_token_ids.append(token_id)
        else:
            skipped_tokens.append(special_tokens[i])

    if local_rank in [-1, 0]:
        print(f"Total special tokens in JSON: {len(special_tokens)}")
        print(f"Valid special tokens found in vocab: {len(valid_special_token_ids)}")
        if skipped_tokens:
            print(f"WARNING: {len(skipped_tokens)} tokens were skipped (UNK): {skipped_tokens[:5]}...")
        if len(valid_special_token_ids) == 0:
            raise ValueError("No valid special tokens found to train! Check your tokenizer.")

    # --- 4. 配置 PEFT ---
    lora_config = TrainableTokensConfig(
        token_indices=valid_special_token_ids,
        target_modules=["embed_tokens"],
        init_weights=True
    )
    
    # 获取 PEFT 模型
    model = get_peft_model(expanded_model, lora_config)
    
    if local_rank in [-1, 0]:
        model.print_trainable_parameters()

    # --- 5. 数据集准备 ---
    # 传入 local_rank 以控制 pandas 日志
    train_dataset = prepare_dataset(train_data_path, local_rank=local_rank)
    val_dataset = prepare_dataset(val_data_path, local_rank=local_rank)

    # 只有主进程显示进度条，或者所有进程都不显示（避免混乱），这里让 HuggingFace map 自动处理
    # map 函数在多进程下，通常建议在主进程处理完缓存，其他进程加载缓存。
    # 但由于 tokenize 比较快，这里为了简单让每个进程独立处理，
    # 或者设置 load_from_cache_file=True 利用 HF 的缓存机制。
    with training_args.main_process_first(desc="Tokenizing dataset"):
        train_dataset = train_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train_data",
            load_from_cache_file=True 
        )
        val_dataset = val_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing val_data",
            load_from_cache_file=True
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # --- 6. 训练 ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    if local_rank in [-1, 0]:
        print("Starting training...")

    trainer.train()

    # --- 7. 评估与保存 ---
    if local_rank in [-1, 0]:
        print("Evaluating...")
        
    result = trainer.evaluate()
    
    if local_rank in [-1, 0]:
        print("Evaluation result:", result)

    # --- 8. 关键修改：多卡环境下的模型合并与保存 ---
    # 等待所有进程完成，防止主进程保存时其他进程还在跑
    trainer.accelerator.wait_for_everyone()

    # 仅在主进程执行 Merge 和 Save
    if local_rank in [-1, 0]:
        print("Merging and unloading model...")
        # 注意：在DDP中，trainer.model 被 wrap 了一层。
        # 但这里的 `model` 变量依然指向内存中的 PEFT 模型对象，
        # 且由于是共享内存或已同步，直接操作是可行的。
        # 为了保险，先从 trainer 获取 unwrapped model
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        
        # 如果 unwrap 后是 PeftModel，执行 merge
        if hasattr(unwrapped_model, "merge_and_unload"):
            merged_model = unwrapped_model.merge_and_unload()
        else:
            # Fallback: 如果 trainer.model 结构复杂，尝试直接用外部变量 model
            # 注意：这要求 model 的权重在训练中被正确原地更新了
            merged_model = model.merge_and_unload()

        output_dir = os.path.join(training_args.output_dir, "best_model")
        print(f"Saving merged model to {output_dir}...")
        
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Save completed successfully.")

if __name__ == "__main__":
    main()