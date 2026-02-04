import torch
from typing import Dict, List, Union
from transformers import PreTrainedTokenizerFast

# 定义预处理函数：只做 Encode，不做 Padding
def preprocess_function(examples, tokenizer, max_seq_length):
    # 这里我们只生成 input_ids，不生成 Tensor，也不 Padding
    return tokenizer(
        examples["prompt"], # 替换为你 json 里的真实文本字段名
        truncation=True,
        max_length=max_seq_length,
        padding=False, # 关键：千万别在这里 Padding，太占空间且不灵活
        return_attention_mask=True
    )

class TrainDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int):
        self.tokenizer = tokenizer
        # max_length 在 preprocess 阶段已经截断过了，这里其实仅仅作为兜底或不需要了
        self.max_length = max_length 

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # features 结构: [{'input_ids': [1, 2], 'attention_mask': [1, 1]}, ...]
        
        # 1. 极速 Padding (C++ 实现，非常快)
        # 这会自动处理 input_ids 和 attention_mask
        batch = self.tokenizer.pad(
            features,
            padding=True,          # 动态 Pad 到当前 batch 最长
            return_tensors="pt"    # 此时转为 Tensor
        )
        
        # 2. 生成 Labels (逻辑和你原来一样)
        batch['labels'] = batch['input_ids'].clone()
        
        # label=-100 让 loss 忽略
        if self.tokenizer.pad_token_id is not None:
            batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
            
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