from dataclasses import dataclass
import os
from pathlib import Path
import json
import sys
import pandas as pd
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

def prepare_dataset(data_path):
    df_j = pd.read_json(data_path, lines=True)
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

if __name__ == "__main__":
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

    expanded_model_path = Path(path_args.expanded_model_path).resolve()
    train_data_path = Path(path_args.train_data_path).resolve()
    val_data_path = Path(path_args.val_data_path).resolve()
    special_tokens_path = Path(path_args.special_tokens_path).resolve()

    expanded_model = AutoModelForCausalLM.from_pretrained(expanded_model_path)
    tokenizer = AutoTokenizer.from_pretrained(expanded_model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    with open(special_tokens_path, "r") as f:
        special_tokens = json.load(f)
        
    tokenized_special_tokens = tokenizer.convert_tokens_to_ids(special_tokens)
    # 进行语义对齐训练
    training_args.label_names = ["labels"]

    lora_config = TrainableTokensConfig(
        token_indices=tokenized_special_tokens,
        target_modules=["embed_tokens"],
        init_weights=True
    )
    model = get_peft_model(expanded_model, lora_config)
    
    train_dataset = prepare_dataset(train_data_path)
    val_dataset = prepare_dataset(val_data_path)

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train_data"
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val_data"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    result = trainer.evaluate()
    print(result)

    # 
    model = model.merge_and_unload()
    output_dir = os.path.join(training_args.output_dir, "best_model")
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    