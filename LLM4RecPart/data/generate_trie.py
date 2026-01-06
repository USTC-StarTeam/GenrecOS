import argparse
import pickle
import json
import os
from transformers import AutoTokenizer
from collections import defaultdict

def load_json(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"文件不存在: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_valid_sids(item_data):
    return [item_info['sid'] for item_info in item_data.values() if item_info.get('sid')]

def generate_trie(input_item_json, tokenizer_path, output_file):
    # 获取合法的 SIDs
    item_data = load_json(input_item_json)
    valid_sids = get_valid_sids(item_data)
    print(f"Found {len(valid_sids)} valid SIDs from item data")
    # 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenized_sids = [tokenizer.encode(sid, add_special_tokens=False) for sid in valid_sids]

    exact_trie = defaultdict(lambda: defaultdict(set))

    for seq in tokenized_sids:
        for pos in range(len(seq)):
            current_token = seq[pos]
            if pos + 1 < len(seq):
                next_token = seq[pos + 1]
                exact_trie[pos][current_token].add(next_token)
            else:
                eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                exact_trie[pos][current_token].add(eos_id)

    final_exact_trie = {}
    for pos in exact_trie:
        final_exact_trie[pos] = {}
        for token_id in exact_trie[pos]:
            final_exact_trie[pos][token_id] = list(exact_trie[pos][token_id])

    for pos in range(6):
        num_tokens = len(final_exact_trie.get(pos, {}))
        print(f"Position {pos}: {num_tokens} possible tokens")

    trie_data = {
        'exact_trie': final_exact_trie,
        'trie_type': 'exact'
    }
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(trie_data, f)
    
    print(f"Exact trie saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_item_json', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)

    args = parser.parse_args()
    generate_trie(args.input_item_json, args.tokenizer_path, args.output_file)