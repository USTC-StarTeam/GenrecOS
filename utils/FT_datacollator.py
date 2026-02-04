import torch
from typing import Dict, List, Union
from transformers import PreTrainedTokenizerFast

# --- 修改版：SFT 专用的预处理函数 ---
def sft_preprocess_function(examples, tokenizer, max_seq_length):
    # 1. 获取 prompt 和 ground_truth
    prompts = examples["prompt"]
    ground_truths = examples["ground_truth"] # 确保 json 里键名是这个
    
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    # 2. 手动遍历拼接，确保 label 对齐
    for prompt, gt in zip(prompts, ground_truths):
        if type(gt) == list:
            gt = "".join(gt)  # 如果 ground_truth 是 list，则拼接成字符串
        # 分别编码，不加特殊 token (我们在最后手动加 EOS)
        # add_special_tokens=False 是为了精准控制拼接
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        gt_ids = tokenizer(gt, add_special_tokens=False)['input_ids']
        
        # 3. 拼接 Input IDs: Prompt + GT
        # 注意：这里假设你的 tokenizer.eos_token_id 存在
        curr_input_ids = prompt_ids + gt_ids
        
        # 4. 构建 Labels: Prompt 部分全是 -100，GT 部分保留
        # -100 是 PyTorch CrossEntropyLoss 的默认 ignore_index
        curr_labels = [-100] * len(prompt_ids) + gt_ids
        
        # 5. 截断处理 (如果超长)
        if len(curr_input_ids) > max_seq_length:
            curr_input_ids = curr_input_ids[-1*max_seq_length:]
            curr_labels = curr_labels[-1*max_seq_length:]
        
        # 6. 生成 Attention Mask
        curr_mask = [1] * len(curr_input_ids)
        
        input_ids_list.append(curr_input_ids)
        labels_list.append(curr_labels)
        attention_mask_list.append(curr_mask)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }

# --- 修改版：支持已生成 Labels 的 Collator ---
class TrainDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length 

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # features 结构: [{'input_ids': [...], 'labels': [-100, -100, 29, ...], 'attention_mask': [...]}, ...]
        
        # 1. 提取 labels，因为 tokenizer.pad 默认不会处理 labels 的 padding (或者会填成 pad_id 而不是 -100)
        labels = [feature['labels'] for feature in features] if 'labels' in features[0] else None
        
        # 2. 处理 input_ids 和 attention_mask 的 padding
        # 将 labels 暂时剔除，避免 tokenizer.pad 报错或错误处理
        features_no_labels = [{k: v for k, v in f.items() if k != 'labels'} for f in features]
        
        batch = self.tokenizer.pad(
            features_no_labels,
            padding=True,
            return_tensors="pt"
        )
        
        # 3. 手动处理 labels 的 padding
        if labels is not None:
            # 找出当前 batch 的最大长度
            max_label_length = max(len(l) for l in labels)
            
            # 创建一个全为 -100 的 tensor
            batch_labels = torch.full((len(labels), max_label_length), -100, dtype=torch.long)
            
            # 填入实际的 label
            for i, label in enumerate(labels):
                batch_labels[i, :len(label)] = torch.tensor(label, dtype=torch.long)
            
            # 赋值回 batch
            batch['labels'] = batch_labels
            
        return batch
    
class EvalDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, features: List[Dict[str, Union[str, List[int]]]]) -> Dict[str, torch.Tensor]:
        # features 结构: [{'input_ids': [...], 'ground_truth': "270,283..."}, ...]

        # 1. 把 ground_truth 单独提取出来，因为它不能被 pad
        groundtruths = []
        features_to_pad = []
        
        for f in features:
            # pop 出来，这样 features_to_pad 里就只剩 input_ids/attention_mask 了
            # 兼容性写法：如果没有 ground_truth 字段就不处理
            if "ground_truth" in f:
                groundtruths.append(f.pop("ground_truth"))
            else:
                groundtruths.append(None)
            if 'user_id' in f:
                f.pop('user_id')  # 移除 user_id，避免 Padding 问题
            features_to_pad.append(f)

        # 2. 极速 Padding
        batch = self.tokenizer.pad(
            features_to_pad,
            padding=True,
            return_tensors="pt"
        )
        
        # 3. 把 ground_truth 放回去 (List[str] 形式)
        # 只要列表不为空，就放回去
        if groundtruths and groundtruths[0] is not None:
            batch['groundtruth'] = groundtruths
        
        return batch