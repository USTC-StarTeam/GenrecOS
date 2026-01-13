import os
import logging
from typing import List, Dict

from datasets import Dataset
from dataclasses import dataclass, field

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO)

@dataclass
class MockTrainingArguments:
    # 路径参数
    output_dir: str = field(
        default="./rq_code_tokenizer",
        metadata={
            "help": "The output directory for the tokenizer.",
            "required": True
        }
    )
    
    # 模型架构参数
    max_length: int = field(
        default=512,
        metadata={
            "help": "The maximum sequence length.",
            "gt": 0
        }
    )
    
    codebook_num: int = field(
        default=3,
        metadata={
            "help": "Number of codebooks. For example: 3 for a, b, c codebooks.",
            "gt": 0,
            "le": 10  # 最多10个codebook
        }
    )
    
    codeword_num_per_codebook: int = field(
        default=20,
        metadata={
            "help": "Number of codewords in each codebook.",
            "gt": 0,
            "le": 1000  # 最多1000个codeword
        }
    )


def build_semantic_id_vocab(
    codebook_num=3,
    codeword_num_per_codebook=20,
    special_tokens_map=None
) -> Dict[str, int]:
    """
    构建Q-Code词汇表
    
    Args:
        codebook_num: codebook数量
        codeword_num_per_codebook: 每个codebook的codeword数量
        special_tokens_map: 特殊token映射，如{"pad_token": "<pad>", ...}
    
    Returns:
        vocab: 词汇表字典 {token: id}
    """
    if special_tokens_map is None:
        special_tokens_map = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]", 
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
        }
    
    # 收集所有特殊token
    special_tokens = [
        special_tokens_map["pad_token"],
        special_tokens_map["unk_token"],
        special_tokens_map["bos_token"], 
        special_tokens_map["eos_token"]
    ]
    
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    next_id = len(special_tokens)
    
    # codebook名称（a, b, c, ...）
    codebook_names = [chr(ord('a') + i) for i in range(codebook_num)]
    
    # 添加codebook token
    for codebook in codebook_names:
        for i in range(1, codeword_num_per_codebook + 1):
            token = f"<{codebook}_{i}>"
            vocab[token] = next_id
            next_id += 1
    
    return vocab


def create_semantic_id_tokenizer(mock_args) -> PreTrainedTokenizerFast:
    """
    创建一个专门处理语义ID编码序列的 Tokenizer。
    例如，处理形如 "<a_101> <b_54> <c_201>" 的字符串。

    Args:
        mock_args: 包含配置参数的对象，需要有：
            - codebook_num: codebook数量
            - codeword_num_per_codebook: 每个codebook的codeword数量
            - max_length: 最大序列长度
            - output_dir: 输出目录

    Returns:
        PreTrainedTokenizerFast: 初始化完成的、可用于 Transformers 的 Tokenizer。
    """
    
    # --- 1. 构建词汇表 ---
    logging.info("Step 1: Building vocabulary from codebook configuration...")
    vocab = build_semantic_id_vocab(
        codebook_num=mock_args.codebook_num,
        codeword_num_per_codebook=mock_args.codeword_num_per_codebook,
        special_tokens_map={
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
        }
    )
    
    logging.info(f"Built vocabulary with {len(vocab)} tokens")
    logging.info(f"Special tokens: [PAD]={vocab['[PAD]']}, [UNK]={vocab['[UNK]']}, "
                 f"[BOS]={vocab['[BOS]']}, [EOS]={vocab['[EOS]']}")
    
    # --- 2. 初始化 Tokenizer 核心 ---
    logging.info("Step 2: Initializing WordLevel tokenizer...")
    # WordLevel 模型非常适合我们的场景，因为它将每个词汇表中的key视为一个独立的单元
    custom_tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))

    # --- 3. 设置预分词器 ---
    logging.info("Step 3: Setting up Whitespace pre-tokenizer...")
    # 我们不再需要复杂的 Regex。新的规则非常简单：用空格分割
    custom_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # --- 4. 设置 Post-Processor ---
    logging.info("Step 4: Setting up post-processor...")
    custom_tokenizer.post_processor = TemplateProcessing(
        single="$A",  # 不自动添加特殊token
        special_tokens=[
            ("[BOS]", vocab["[BOS]"]),
            ("[EOS]", vocab["[EOS]"]),
        ],
    )
    # 这里是不自动添加，可通过tokenizer.encode("<a_101> <b_42>", add_special_tokens=True) tokenizer(["<a_101> <b_42>", "<c_999>"]， add_special_tokens=False（注意默认是True）)  手动加上

    # --- 5. 配置 Padding ---
    logging.info("Step 5: Enabling padding...")
    custom_tokenizer.enable_padding(
        direction="left",
        pad_id=vocab["[PAD]"],
        pad_token="[PAD]",
        length=mock_args.max_length,
    )
    
    # --- 6. 保存并包装 ---
    logging.info("Step 6: Saving tokenizer...")
    os.makedirs(mock_args.output_dir, exist_ok=True)
    tokenizer_file_path = os.path.join(mock_args.output_dir, "tokenizer.json")
    custom_tokenizer.save(tokenizer_file_path)
    
    # --- 7. 创建 PreTrainedTokenizerFast ---
    logging.info("Step 7: Creating PreTrainedTokenizerFast wrapper...")
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file_path,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=mock_args.max_length,
        padding_side="left",
    )
    logging.info("Tokenizer creation complete!")
    return hf_tokenizer


# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 创建tokenizer
    mock_args = MockTrainingArguments(
        output_dir="./test_tokenizer",
        max_length=20,
        codebook_num=3,
        codeword_num_per_codebook=10
    )
    
    tokenizer = create_semantic_id_tokenizer(mock_args)
    
    print("Tokenizer创建成功!")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    
    # 2. 测试编码/解码
    test_texts = [
        "<a_1>",
        "<a_1> <b_2> <c_3>",
        "<a_10> <b_5> <c_1>",
    ]
    
    print("\n编码/解码测试:")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"  '{text}' -> {encoded} -> '{decoded}'")
        
        # 验证
        if decoded == text:
            print(f"    ✅ 正确")
        else:
            print(f"    ❌ 错误")
    
    # 3. 测试特殊token
    print("\n特殊token测试:")
    print(f"  [PAD]: ID={tokenizer.pad_token_id}")
    print(f"  [UNK]: ID={tokenizer.unk_token_id}")
    print(f"  [BOS]: ID={tokenizer.bos_token_id}")
    print(f"  [EOS]: ID={tokenizer.eos_token_id}")
    
    # 4. 测试padding
    print("\nPadding测试:")
    batch = ["<a_1> <b_2>", "<c_3> <a_5> <b_7> <c_2>"]
    encoded = tokenizer(batch, padding=True, return_tensors="pt")
    print(f"  批量输入: {batch}")
    print(f"  输入IDs形状: {encoded['input_ids'].shape}")
    print(f"  Padding后序列: {encoded['input_ids'][0].tolist()}")
    
    # 5. 测试未知token
    print("\n未知token测试:")
    unknown_text = "<a_1> <x_99> <b_2>"
    encoded = tokenizer.encode(unknown_text)
    print(f"  '{unknown_text}' -> {encoded}")
    
    if tokenizer.unk_token_id in encoded:
        print(f"    ✅ 未知token正确处理")
    else:
        print(f"    ❌ 未知token处理可能有问题")
    
    print("\n✅ 所有快速测试通过!")
