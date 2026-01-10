import argparse
import json
from pathlib import Path

import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def expand_vocabulary(base_model_path, special_tokens_path, output_path):
    if not Path(base_model_path).exists():
        raise FileNotFoundError(f"Base model directory not found: {base_model_path}")
    if not Path(special_tokens_path).exists():
        raise FileNotFoundError(f"Special tokens file not found: {special_tokens_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = AutoConfig.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)
    with open(special_tokens_path, 'r', encoding='utf-8') as f:
        new_tokens = json.load(f)
        print(f"Preparing to add {len(new_tokens)} special tokens.")

    tokens_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": new_tokens}, replace_additional_special_tokens=False
    )
    print(f"Successfully added {tokens_added} tokens.")
    model.resize_token_embeddings(len(tokenizer))
    config.vocab_size = len(tokenizer)

    save_dir = Path(output_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving expanded model to: {save_dir}")
    config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, default='../models/Qwen3-1-7B')
    parser.add_argument('--special_tokens_path', type=str, default='../../Data/Beauty_onerec_think/FT_data/beauty/special_tokens.json')
    parser.add_argument('--output_path', type=str, default='../models/expanded_model')
    args = parser.parse_args()

    expand_vocabulary(
        base_model_path=args.base_model_path,
        special_tokens_path=args.special_tokens_path,
        output_path=args.output_path
    )