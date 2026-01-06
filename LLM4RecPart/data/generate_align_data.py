import argparse
import json
import os
import pandas as pd

def load_json(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"文件不存在: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_valid_item_ids(item_ids, item_data):
    valid_ids = [
        id for id in item_ids 
        if id in item_data and item_data[id]['sid'] and item_data[id]['title'] and item_data[id]['categories']
    ]
    return valid_ids

def generate_item_descs(valid_item_ids, item_data):
    descriptions = []
    for item_id in valid_item_ids:
        item_info = item_data.get(item_id)
        sid = item_info.get("sid")
        title = item_info.get("title")
        categories = item_info.get("categories")
        item_desc = f'{sid}, its title is "{title}", its categories are "{categories}"'
        descriptions.append(item_desc)
    return descriptions

def format_user_description(item_descs):
    description = "The user has purchased the following items: " + "; ".join(item_descs) + ";"
    return description

def generate_align_data(input_item_json, input_seq_json, output_dir):
    # 读取物品信息
    item_data = load_json(input_item_json)
    print(f"Loaded {len(item_data)} item descriptions from {input_item_json}")
    # 读取用户行为序列数据
    seq_data = load_json(input_seq_json)
    print(f"Loaded {len(seq_data)} user sequences from {input_seq_json}")

    # 对齐数据
    align_data_train = []
    align_data_val = []
    align_data_test = []
    for user_id, item_ids in seq_data.items():
        # 筛选出合法的 item id
        
        seq_len = len(item_ids)
        if seq_len > 0:
            valid_ids = get_valid_item_ids(item_ids, item_data)
            if valid_ids:
                item_descs = generate_item_descs(valid_ids, item_data)
                test_description = format_user_description(item_descs)
                align_data_test.append({
                    'user_id': user_id,
                    'description': test_description
                })
        if seq_len > 1:
            valid_ids = get_valid_item_ids(item_ids[:-1], item_data)
            if valid_ids:
                item_descs = generate_item_descs(valid_ids, item_data)
                val_description = format_user_description(item_descs)
                align_data_val.append({
                    'user_id': user_id,
                    'description': val_description
                })
        if seq_len > 2:
            valid_ids = get_valid_item_ids(item_ids[:-2], item_data)
            if valid_ids:
                item_descs = generate_item_descs(valid_ids, item_data)
                train_description = format_user_description(item_descs)
                if train_description:
                    align_data_train.append({
                    'user_id': user_id,
                    'description': train_description
                })
    print(f"Generated align data - Train: {len(align_data_train)}, Val: {len(align_data_val)}, Test: {len(align_data_test)}")
    # 保存对齐数据
    df_train = pd.DataFrame(align_data_train)
    df_val = pd.DataFrame(align_data_val)
    df_test = pd.DataFrame(align_data_test)
    os.makedirs(output_dir, exist_ok=True)
    df_train.to_json(os.path.join(output_dir, 'train.json'), orient='records', lines=True, force_ascii=False)
    df_val.to_json(os.path.join(output_dir, 'val.json'), orient='records', lines=True, force_ascii=False)
    df_test.to_json(os.path.join(output_dir, 'test.json'), orient='records', lines=True, force_ascii=False)
    print(f"Align data saved to directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_item_json', type=str, required=True)
    parser.add_argument('--input_seq_json', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    generate_align_data(args.input_item_json, args.input_seq_json, args.output_dir)