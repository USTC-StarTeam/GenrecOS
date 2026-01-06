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

