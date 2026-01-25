import torch
from typing import Dict, List, Union
from transformers import PreTrainedTokenizerFast
# # 训练数据整理器
# class TrainDataCollator:
#     def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int):
#         """
#         训练数据的整理器。

#         Args:
#             tokenizer (PreTrainedTokenizerFast): 使用的 tokenizer。
#             max_length (int): 最大序列长度。
#             item_token_length (int): 每个物品由多少个 token 组成 (例如: <a_id> <b_id> <c_id> 是 3 个 token)。
#         """
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
#         # 模型输入
#         sequences = [e["history"] for e in examples]
#         # 对输入序列进行编码
#         batch_dict = self.tokenizer(
#             sequences,
#             is_split_into_words=False,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
        
#         batch_dict['labels'] = batch_dict['input_ids'].clone()
        
#         # label=-100会让loss忽略
#         if self.tokenizer.pad_token_id is not None:
#             batch_dict["labels"][batch_dict["labels"] == self.tokenizer.pad_token_id] = -100
        
#         return batch_dict

# # 评估数据整理器
# class EvalDataCollator:
#     def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int):
#         """
#         评估数据的整理器。

#         Args:
#             tokenizer (PreTrainedTokenizerFast): 使用的 tokenizer。
#             max_length (int): 最大序列长度。
#         """
#         self.tokenizer = tokenizer
#         self.max_length = max_length
        
#     def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]) -> Dict[str, torch.Tensor]:
#         # 模型输入
#         sequences = [e["history"] for e in examples]

#         # 对输入序列进行编码
#         batch_dict = self.tokenizer(
#             sequences,
#             is_split_into_words=False,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
        
#         # 提取 Ground Truth Item ID
#         # groundtruth 字段是 Item ID 字符串，例如 "270,283,393"
#         # 这是一个多对一的评估，每个用户有多个 ground truth
#         groundtruths = [e["ground_truth"] for e in examples]
#         # 将 groundtruth 转换为 Item ID 列表，并加入 batch_dict 以便传递给 compute_metrics
#         batch_dict['groundtruth'] = groundtruths
        
#         # 注意: 在评估阶段，我们不使用 Trainer 的 compute_loss，而是使用 generate()
#         # 因此 'labels' 字段不是必须的
#         return batch_dict

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