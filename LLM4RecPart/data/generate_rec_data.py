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
        if id in item_data and item_data[id]['sid']
    ]
    return valid_ids

def generate_item_descs(valid_item_ids, item_data):
    descriptions = []
    for item_id in valid_item_ids:
        item_info = item_data.get(item_id)
        sid = item_info.get("sid")
        descriptions.append(sid)
    return descriptions

def format_user_description(item_descs):
    description = "The user has purchased the following items: " + "; ".join(item_descs) + ";"
    return description

def generate_rec_data(input_item_json, input_seq_json, output_dir):
    # 读取物品信息
    item_data = load_json(input_item_json)
    print(f"Loaded {len(item_data)} item descriptions from {input_item_json}")
    # 读取用户行为序列数据
    seq_data = load_json(input_seq_json)
    print(f"Loaded {len(seq_data)} user sequences from {input_seq_json}")

    # 推荐数据
    rec_data_train = []
    rec_data_val = []
    rec_data_test = []
    for user_id, item_ids in seq_data.items():
        # 筛选出合法的 item id
        valid_ids = get_valid_item_ids(item_ids, item_data)
        seq_len = len(valid_ids)

        # 生成物品描述列表
        item_descs = generate_item_descs(valid_ids, item_data)
        if seq_len > 0:
            test_description = format_user_description(item_descs[:-1])
            rec_data_test.append({
                'user_id': user_id,
                'description': test_description,
                "groundtruth": [item_data[valid_ids[-1]]['sid']]
            })
        if seq_len > 1:
            val_description = format_user_description(item_descs[:-2])
            rec_data_val.append({
                'user_id': user_id,
                'description': val_description,
                "groundtruth": [item_data[valid_ids[-2]]['sid']]
            })
        if seq_len > 2:
            train_description = format_user_description(item_descs[:-3])
            rec_data_train.append({
                'user_id': user_id,
                'description': train_description,
                "groundtruth": [item_data[valid_ids[-3]]['sid']]
            })
    print(f"Generated rec data - Train: {len(rec_data_train)}, Val: {len(rec_data_val)}, Test: {len(rec_data_test)}")
    # 保存推荐数据
    df_train = pd.DataFrame(rec_data_train)
    df_val = pd.DataFrame(rec_data_val)
    df_test = pd.DataFrame(rec_data_test)
    os.makedirs(output_dir, exist_ok=True)
    df_train.to_json(os.path.join(output_dir, 'train.json'), orient='records', lines=True, force_ascii=False)
    df_val.to_json(os.path.join(output_dir, 'val.json'), orient='records', lines=True, force_ascii=False)
    df_test.to_json(os.path.join(output_dir, 'test.json'), orient='records', lines=True, force_ascii=False)
    print(f"Rec data saved to directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_item_json', type=str, required=True)
    parser.add_argument('--input_seq_json', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    generate_rec_data(args.input_item_json, args.input_seq_json, args.output_dir)