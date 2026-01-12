import os
import json
import logging
import yaml
import argparse
import sys
import tempfile
import warnings
from typing import List, Dict, Union
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    PreTrainedTokenizerFast,
    AddedToken,
    Qwen2Tokenizer,
    LogitsProcessorList,
)
import transformers.utils.logging
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# 导入你的自定义模型代码
from llamarec import LlamaRecForCausalLM, LlamaRecConfig

# 导入同事写的工具代码
from util.datacollator import TrainDataCollator, EvalDataCollator
from util.utils_evaluate import (
    build_item_token_codebooks_dynamically, 
    beamsearch_prefix_constraint_fn, 
    DynamicHierarchicalLogitsProcessor,
)
from util.eval import compute_hr_at_k, compute_ndcg_at_k

# 忽略特定的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.trainer")

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

    # step 5: 保存结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Tokenizer saved to: {output_dir}")
    logging.info(f"Final vocab size: {len(tokenizer)}")

    return tokenizer

# 包含生成式评估的训练流程
class CustomTrainer(Trainer):
    def __init__(self, eval_collator, generation_config_params, **kwargs):
        super().__init__(**kwargs)
        self.eval_collator = eval_collator
        # 将生成需要的参数存下来
        self.gen_len = generation_config_params['generation_length']
        self.num_beams = generation_config_params['num_beams']
        self.k_values = generation_config_params['k_values']
        self.item_token_codebooks = generation_config_params['item_token_codebooks']

        # --- 【极速优化】构建 NumPy 向量化查找表 ---
        vocab = kwargs['processing_class'].get_vocab()
        
        # 1. 找到最大的 ID，确定数组大小
        max_id = max(vocab.values())
        
        # 2. 初始化一个 object 类型的数组，默认填空字符串 ""
        # 使用 dtype=object 是因为我们的 token 字符串长度不固定
        self.vocab_array = np.array(["" for _ in range(max_id + 1)], dtype=object)
        
        # 3. 填充数组：index 就是 ID，value 就是 token 字符串
        for k, v in vocab.items():
            self.vocab_array[v] = k
            
        logging.info("✅ NumPy Vectorized Vocab Lookup Table built.")

    # 重写 evaluate 方法以支持生成指标 (HR/NDCG)
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        
        # 1. 准备 DataLoader
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.eval_collator, 
            shuffle=False,
            drop_last=False
        )

        # 2. 准备模型
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()
        
        logging.info(f"***** Running Custom Evaluation (Generation) *****")
        
        total_metrics_sum = {f"HR@{k}": 0.0 for k in self.k_values}
        total_metrics_sum.update({f"NDCG@{k}": 0.0 for k in self.k_values})
        total_samples = 0

        
        # 3. 循环生成
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(eval_dataloader)): # 加个 enumerate 方便看是第几个 batch
                # --- 计时起点 ---
                torch.cuda.synchronize()
                
                # 1. 数据移动
                input_ids = batch['input_ids'].to(self.args.device)
                attention_mask = batch['attention_mask'].to(self.args.device)
                groundtruth = batch['groundtruth'] # List[str]

                batch_size = input_ids.shape[0]
                prompt_length = input_ids.shape[1]

                torch.cuda.synchronize()

                # 实例化我们新的 Processor
                # 注意：必须放在循环里，因为 prompt_length 可能会随 batch 变化
                logits_processor = LogitsProcessorList([
                    DynamicHierarchicalLogitsProcessor(
                        prompt_length=prompt_length,
                        item_token_codebooks=self.item_token_codebooks,
                        device=self.args.device
                    )
                ])

                # 2. 模型生成
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=prompt_length + self.gen_len,
                    num_beams=self.num_beams,
                    do_sample=False,
                    num_return_sequences=self.num_beams,
                    pad_token_id=self.processing_class.pad_token_id,
                    eos_token_id=self.processing_class.eos_token_id,
                    # prefix_allowed_tokens_fn=batch_beamsearch_prefix_constraint_fn,
                    logits_processor=logits_processor, 
                    use_cache=True
                )
                
                torch.cuda.synchronize() # 等待 GPU 生成完毕

                # # 3. 解码 (CPU 字符串操作，如果 Batch 很大这里会慢)
                # new_tokens = generated_ids[:, -self.gen_len:]
                # predicted_token_sequences = self.processing_class.batch_decode(new_tokens, skip_special_tokens=False)
                
                # 1. 搬运到 CPU 并转为 numpy (O(1)耗时)
                # shape: [Batch_Size * Num_Beams, Gen_Len]
                new_tokens_cpu = generated_ids[:, -self.gen_len:].cpu().numpy()
                
                # 2. 向量化查表 (Instant Lookup)
                # 直接用 ID 数组作为索引，瞬间得到对应的字符串数组
                # shape: [N, Gen_Len]，内容变成了 ["<a_1>", "<b_2>", ...]
                token_strs = self.vocab_array[new_tokens_cpu]
                
                # 3. 向量化拼接 (Vectorized Join)
                # 既然 Gen_Len 通常很短（比如3或4），我们直接按列相加
                # NumPy 的 object array 支持用 + 号进行字符串拼接，这比 Python 循环快得多
                
                if self.gen_len == 1:
                    predicted_token_sequences = token_strs.flatten().tolist()
                else:
                    # 这是一个累加过程：Col0 + Col1 + Col2 ...
                    # 比如 ["<a_1>"] + ["<b_1>"] = ["<a_1><b_1>"]
                    # 这种操作是在 C 层面循环的
                    result_array = token_strs[:, 0]
                    for i in range(1, self.gen_len):
                        result_array = result_array + token_strs[:, i]
                    
                    predicted_token_sequences = result_array.tolist()

                # 4. Reshape & 指标计算 (纯 CPU 逻辑)
                reshaped_token_sequences = [
                    predicted_token_sequences[i : i + self.num_beams]
                    for i in range(0, len(predicted_token_sequences), self.num_beams)
                ]

                batch_hr = compute_hr_at_k(reshaped_token_sequences, groundtruth, self.k_values)
                batch_ndcg = compute_ndcg_at_k(reshaped_token_sequences, groundtruth, self.k_values)

                # 5. 累加
                for k_val in self.k_values:
                    total_metrics_sum[f"HR@{k_val}"] += batch_hr[f"HR@{k_val}"] * batch_size
                    total_metrics_sum[f"NDCG@{k_val}"] += batch_ndcg[f"NDCG@{k_val}"] * batch_size
                
                total_samples += batch_size

        # 4. 汇总指标
        metrics = {f"{metric_key_prefix}_{k}": (v / total_samples) for k, v in total_metrics_sum.items()}
        
        # 5. 记录日志
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        
        logging.info(f"Evaluation metrics: {metrics}")
        return metrics


# =============================================================================
# 3. 整合后的 Main 函数
# =============================================================================
def main():
    # 获取配置文件路径
    parser = argparse.ArgumentParser(description="Train a LlamaRec model using a YAML config file.")
    parser.add_argument("--dataset", type=str, default='Beauty')
    parser.add_argument("--model_name", type=str, default='llama-rec')
    args = parser.parse_args()

    # 1. 稳健的路径读取
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_script_dir, "pretrain_config", f"{args.dataset}_{args.model_name}.yaml")
    
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # 从解析的文件中提取配置
    paths_config = config_data['paths']
    model_params = config_data['model_params']
    training_args_dict = config_data['training_args']
    tokenizer_params = config_data['tokenizer_params']
    testing_args = config_data['testing_args']

    # 路径与参数处理
    # 假设 dataset_path 指向包含 train.json/test.json 的目录
    dataset_path = paths_config['dataset_path'] 
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(paths_config['output_dir'], f"{model_params.get('MODEL_TYPE', 'model')}_{timestamp}")
    tokenizer_dir = paths_config['tokenizer_dir']
    max_seq_length = model_params['max_seq_length']
    # 注意：codebook_num 对应的就是 generation 的长度（每个 item 有几层 ID）
    # 这里假设 YAML 里写的是 codeword_nums 列表，codebook_num 是列表长度
    codeword_nums = tokenizer_params.get('codeword_nums', [20, 20, 20])
    generation_length = len(codeword_nums)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ================= 日志配置区域 =================
    log_file_path = os.path.join(output_dir, "training_process.log")
    
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    transformers_logger = transformers.utils.logging.get_logger("transformers")
    transformers_logger.addHandler(file_handler)
    
    logging.info(f"✅ Logging started. Output file: {log_file_path}")
    # ==========================================================

    # ==========================================================
    # Tokenizer 创建 (替换为你的 create_pure_id_qwen_tokenizer)
    # ==========================================================
    # tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
    
    # 逻辑：如果没有现成的 json，或者为了保证配置一致，建议使用 create_pure_id_qwen_tokenizer
    # 它内部是基于内存构建的 Qwen2Tokenizer，非常轻量
    
    # 只要 YAML 里配了 codeword_nums，我们就动态构建，确保一致性
    tokenizer = create_pure_id_qwen_tokenizer(
        output_dir=tokenizer_dir,
        codeword_nums=codeword_nums
    )

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

    # ==========================================================
    # 数据集加载
    # ==========================================================
    # # 假设目录结构是 train_data.json 和 test_data.json
    # data_files_train = os.path.join(dataset_path, 'train_data.json')
    # data_files_test = os.path.join(dataset_path, 'test_data.json') # 或者是 valid
    
    # 使用 data_files 参数加载指定文件
    train_dataset = load_dataset("json", data_dir=dataset_path, split='train')
    # 如果没有单独的 test 文件，用 train 切分或者怎样，这里假设有
    # 同事代码里用的也是 data_files=dataset_path（可能是个包含多个json的目录？），这里按标准写法
    try:
        eval_dataset = load_dataset("json", data_dir=dataset_path, split='valid')
    except:
        logging.warning("Test file not found, using train set as eval (FOR DEBUG ONLY)")
        eval_dataset = train_dataset

    # ==========================================================
    # 模型构建
    # ==========================================================
    logging.info("Creating model from scratch...")
    
    config = LlamaRecConfig(
        # 核心：将 Tokenizer 的实际大小传给 Config
        vocab_size=len(tokenizer),
        
        hidden_size=model_params['hidden_size'],
        intermediate_size=model_params['intermediate_size'],
        num_hidden_layers=model_params['num_hidden_layers'],
        num_attention_heads=model_params['num_attention_heads'],
        # 序列长度 = Prompt 长度 + 生成长度
        max_position_embeddings=max_seq_length + generation_length,
        rms_norm_eps=model_params['rms_norm_eps'],
        model_type=model_params.get('MODEL_TYPE', 'llama-rec'),
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        
        # 这些是 LlamaRec 特有的参数，如果 config 里有就传
        # num_levels=len(codeword_nums), 
    )
    model = LlamaRecForCausalLM(config)
    logging.info(f"Model created with {model.num_parameters() / 1e6:.2f} M parameters.")

    # ==========================================================
    # Trainer 准备
    # ==========================================================
    training_args_dict['output_dir'] = output_dir
    training_args_dict['logging_dir'] = os.path.join(output_dir, 'logs')
    training_args = TrainingArguments(**training_args_dict)

    # DataCollator
    # 注意：确保 Collator 里的 tokenizer 调用参数是正确的（is_split_into_words=False）
    train_collator = TrainDataCollator(tokenizer=tokenizer, max_length=max_seq_length)
    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=max_seq_length)

    # 动态构建 Codebooks (用于生成约束)
    # 这需要利用你的 tokenizer 来解析 <a_0> 对应的 ID
    item_token_codebooks = build_item_token_codebooks_dynamically(tokenizer, generation_length)
    
    generation_config_params = {
        "generation_length": generation_length,
        "num_beams": testing_args['num_beams'],
        "k_values": testing_args['eval_k_values'],
        "item_token_codebooks": item_token_codebooks
    }

    # 实例化 CustomTrainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer, # 传入 tokenizer 对象
        data_collator=train_collator,
        eval_collator=eval_collator,
        generation_config_params=generation_config_params,
        # 早停策略
        callbacks=[EarlyStoppingCallback(early_stopping_patience=testing_args['early_stopping_patience'])] 
    )

    # ==========================================================
    # 训练与保存
    # ==========================================================
    logging.info("Starting training...")
    trainer.train()

    # 打印最优结果
    if trainer.state.best_model_checkpoint:
        best_metric = training_args.metric_for_best_model
        logging.info("=" * 40)
        logging.info(f"🏆 Best Model Checkpoint: {trainer.state.best_model_checkpoint}")
        logging.info(f"Best Metric ({best_metric}): {trainer.state.best_metric}")
        logging.info("=" * 40)
    
    # 保存最终模型 (Best Model)
    # 如果 load_best_model_at_end=True，此时 model 已经是最好的了
    final_model_path = os.path.join(output_dir, "best_model")
    logging.info(f"Saving best model to {final_model_path}")
    
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logging.info("All operations complete!")

if __name__ == "__main__":
    main()