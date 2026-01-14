import os
import logging
import json
from typing import Dict, List, Union, Optional, Any, Set
from datasets import Dataset
import torch
import numpy as np

from transformers import (
    PreTrainedTokenizerFast,
)
logging.basicConfig(level=logging.INFO)


def build_item_token_codebooks_dynamically(
    tokenizer: PreTrainedTokenizerFast,
    codebook_num: int 
) -> List[List[int]]:
    """
    根据 codebook_num 动态生成 ['<a_', '<b_', ...] 前缀列表，并从 tokenizer 
    的词汇表中构造独立的约束码本列表。

    Args:
        tokenizer (PreTrainedTokenizerFast): 包含自定义前缀 tokens 的分词器。
        codebook_num (int): 约束的步长数（码本的数量，例如 3 或 4）。

    Returns:
        List[List[int]]: 包含对应前缀 Token ID 集合的列表。
    """
        
    # 动态生成前缀列表
    prefix_list = []
    start_char_code = ord('a')
    
    for i in range(codebook_num):
        char = chr(start_char_code + i)
        prefix_list.append(f"<{char}_")
    
    # 从词汇表提取 Token ID
    vocab: Dict[str, int] = tokenizer.get_vocab()
    generation_length = codebook_num
    
    # 初始化码本列表 (长度为 generation_length)
    item_token_codebooks: List[List[int]] = [[] for _ in range(generation_length)]
    
    # 遍历词汇表，根据动态生成的前缀进行填充
    for token, token_id in vocab.items():
        for i, prefix in enumerate(prefix_list):
            if token.startswith(prefix):
                item_token_codebooks[i].append(token_id)
                break
    # 排序并返回
    for codebook in item_token_codebooks:
        codebook.sort()

    return item_token_codebooks

# 这里是之前欣锐写的beamsearch规范输出情况的代码，但是太慢了
def beamsearch_prefix_constraint_fn(
    batch_id: int, 
    input_ids_tensor: torch.Tensor,
    prompt_length: int, 
    generation_length: int,
    item_token_codebooks: List[List[int]]
) -> List[int]:
    """
    根据当前已生成的 token 数量，返回对应的约束码本。
    """
    current_length = input_ids_tensor.size(-1)
    L = current_length - prompt_length 

    if L < generation_length:
        # L < L_gen: 返回约束码本
        return item_token_codebooks[L]
        
    elif L == generation_length:
        # L = L_gen: 已经生成了足够的 tokens，强制停止！
        return [] # 返回空列表，确保模型在下一步无法选择任何 token
        
    else: # L > generation_length
        return []

# 于是拼尽全力（让Gemini）写了个新的
from transformers import LogitsProcessor
class DynamicHierarchicalLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_length: int, item_token_codebooks: List[List[int]], device):
        """
        Args:
            prompt_length: Prompt 的长度
            item_token_codebooks: 目前的码本 [[Codebook_A_IDs], [Codebook_B_IDs], ...]
            device: GPU 设备
        """
        self.prompt_length = prompt_length
        self.codebook_len = len(item_token_codebooks)
        
        # 【关键优化】
        # 将 Python List 预先转为 GPU 上的 Tensor，避免在生成循环中发生 CPU->GPU 拷贝
        # 即使未来是前缀树，也应该把树结构存为 Tensor 形式
        self.allowed_tokens_tensors = [
            torch.tensor(ids, device=device, dtype=torch.long) 
            for ids in item_token_codebooks
        ]
        
        # 预先创建一个 -inf 的标量，用于填充 Mask
        self.neg_inf = float('-inf')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids shape: [Batch_Size * Num_Beams, Current_Seq_Len]
        # scores shape:    [Batch_Size * Num_Beams, Vocab_Size]

        # 1. 计算当前是生成的第几步 (step)
        # 例如 prompt=10, current_len=10 -> step=0 (生成第一个token)
        current_step = input_ids.shape[1] - self.prompt_length
        
        # 2. 如果超出了约束范围，不做限制
        if current_step >= self.codebook_len:
            return scores

        # =================================================================
        # 【未来扩展点：动态前缀树 Trie】
        # 将来你可以在这里插入逻辑：
        # last_token_ids = input_ids[:, -1] # 获取上一步生成的 ID
        # next_allowed_ids = MyTrie.lookup(last_token_ids) # 查表得到下一步允许的 ID
        # mask = ... 构建 mask ...
        # =================================================================
        
        # 3. 当前逻辑：按步长获取允许的 ID Tensor (GPU 操作)
        allowed_ids = self.allowed_tokens_tensors[current_step]
        
        # 4. 构建 Mask (全向量化操作，无 Python 循环)
        # 方法：创建一个全 -inf 的 mask，然后把允许的位置填 0
        
        # 4.1 创建一个与 scores 形状相同的 mask，初始全为 -inf
        # torch.full_like 很快，因为它是在 GPU 上直接分配
        mask = torch.full_like(scores, self.neg_inf)
        
        # 4.2 将允许的列（Token ID）设为 0
        # index_fill_ 是原地操作，速度极快
        # 这里的逻辑是：对所有 Batch，允许的 Token 都是一样的 (allowed_ids)
        # 如果将来每个 Batch 允许的不一样，这里可以用 scatter_ 操作
        mask.index_fill_(1, allowed_ids, 0)
        
        # 5. 应用 Mask
        return scores + mask