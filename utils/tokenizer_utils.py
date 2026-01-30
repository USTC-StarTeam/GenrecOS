import tempfile
from typing import List
import os
import json
import logging
from transformers import (
    PreTrainedTokenizerFast,
    AddedToken,
    Qwen2Tokenizer,
)

# 纯净版 Qwen Tokenizer 构建函数
def create_pure_id_qwen_tokenizer(
    output_dir: str, 
    codeword_nums: List[int]  # e.g., [100, 200, 400]
):
    """
    基于 Qwen2Tokenizer 源码，从零构建一个纯净的、只包含语义 ID 的分词器。
    """
    logging.info(f"Building Pure ID Qwen Tokenizer with codeword_nums={codeword_nums}...")
    
    # step 1: 准备一个极简的 Dummy 词表
    dummy_vocab = {"<|endoftext|>": 0}
    
    # 使用临时目录生成这两个必须的文件
    with tempfile.TemporaryDirectory() as temp_dir:
        vocab_file = os.path.join(temp_dir, "vocab.json")
        merges_file = os.path.join(temp_dir, "merges.txt")
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(dummy_vocab, f)
        with open(merges_file, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n") 
            
        # step 2: 初始化原生 Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            unk_token="<|endoftext|>",
            pad_token="<|endoftext|>",
            bos_token=None, 
            eos_token="<|endoftext|>",
        )

    # step 3: 构建你的语义 ID (AddedToken)
    new_tokens = []
    
    # 推荐系统常用的控制符
    control_tokens = [
        AddedToken("[PAD]", special=True, normalized=False),
        AddedToken("[MASK]", special=True, normalized=False),
    ]
    new_tokens.extend(control_tokens)
    
    # 生成语义 ID <a_0>, <b_10> ...
    for i, count in enumerate(codeword_nums):
        prefix = chr(ord('a') + i)
        for j in range(count):
            token_content = f"<{prefix}_{j}>"
            # 核心配置：special=True 启用 Trie 树贪婪匹配，解决无空格分词问题
            new_tokens.append(AddedToken(
                token_content, 
                special=True, 
                normalized=False, 
                lstrip=False, 
                rstrip=False
            ))

    # step 4: 注入 Token
    logging.info(f"Injecting {len(new_tokens)} semantic tokens into tokenizer...")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": new_tokens}, 
        replace_additional_special_tokens=False
    )
    
    # 更新 pad_token_id
    if "[PAD]" in tokenizer.get_vocab():
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")


    # 修改一下tokenizer的padding位置
    tokenizer.padding_side = "left"   # 强制设为左填充
    tokenizer.truncation_side = "left" # (可选) 截断通常也设为左侧，保留最新的历史

    # 健壮性检查
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    # Qwen 默认无 BOS/EOS，这里用 <|endoftext|> 或者我们刚加的 [PAD] 兜底，或者根据模型逻辑指定
    # 如果你的模型依赖 BOS/EOS 启动/结束，确保它们存在
    if tokenizer.bos_token_id is None: 
         # 如果词表里没 [BOS]，用 <|endoftext|> 顶替
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if tokenizer.eos_token_id is None: 
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    logging.info(f"Final check - vocab: {len(tokenizer)}, pad: {tokenizer.pad_token_id}, bos: {tokenizer.bos_token_id}, eos: {tokenizer.eos_token_id}")

    # step 5: 保存结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Tokenizer saved to: {output_dir}")
    logging.info(f"Final vocab size: {len(tokenizer)}")

    return tokenizer