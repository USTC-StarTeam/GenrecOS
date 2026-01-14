import torch
from typing import Dict, List, Union
from transformers import PreTrainedTokenizerFast
# 训练数据整理器
class TrainDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int):
        """
        训练数据的整理器。

        Args:
            tokenizer (PreTrainedTokenizerFast): 使用的 tokenizer。
            max_length (int): 最大序列长度。
            item_token_length (int): 每个物品由多少个 token 组成 (例如: <a_id> <b_id> <c_id> 是 3 个 token)。
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # 模型输入
        sequences = [e["history"] for e in examples]
        # 对输入序列进行编码
        batch_dict = self.tokenizer(
            sequences,
            is_split_into_words=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        batch_dict['labels'] = batch_dict['input_ids'].clone()
        
        # label=-100会让loss忽略
        if self.tokenizer.pad_token_id is not None:
            batch_dict["labels"][batch_dict["labels"] == self.tokenizer.pad_token_id] = -100
        
        return batch_dict

# 评估数据整理器
class EvalDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int):
        """
        评估数据的整理器。

        Args:
            tokenizer (PreTrainedTokenizerFast): 使用的 tokenizer。
            max_length (int): 最大序列长度。
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]) -> Dict[str, torch.Tensor]:
        # 模型输入
        sequences = [e["history"] for e in examples]

        # 对输入序列进行编码
        batch_dict = self.tokenizer(
            sequences,
            is_split_into_words=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 提取 Ground Truth Item ID
        # groundtruth 字段是 Item ID 字符串，例如 "270,283,393"
        # 这是一个多对一的评估，每个用户有多个 ground truth
        groundtruths = [e["ground_truth"] for e in examples]
        # 将 groundtruth 转换为 Item ID 列表，并加入 batch_dict 以便传递给 compute_metrics
        batch_dict['groundtruth'] = groundtruths
        
        # 注意: 在评估阶段，我们不使用 Trainer 的 compute_loss，而是使用 generate()
        # 因此 'labels' 字段不是必须的
        return batch_dict