import os
import logging
from typing import Dict, List, Union, Optional, Any
from datasets import load_dataset, Dataset
from functools import partial
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import argparse

# 导入你的自定义代码
from llamarec import create_semantic_id_tokenizer, MockTrainingArguments 
from llamarec import LlamaRecForCausalLM, LlamaRecConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
logging.basicConfig(level=logging.INFO)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
        sequences = [e["text"] for e in examples]
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
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_length: int, item_token_length: int = 3):
        """
        评估数据的整理器。

        Args:
            tokenizer (PreTrainedTokenizerFast): 使用的 tokenizer。
            max_length (int): 最大序列长度。
            item_token_length (int): 每个物品由多少个 token 组成 (例如: <a_id> <b_id> <c_id> 是 3 个 token)。
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.item_token_length = item_token_length 
        
    def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]) -> Dict[str, torch.Tensor]:
        # 模型输入
        sequences = [e["text"] for e in examples]

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
        batch_dict['groundtruth_item_ids'] = groundtruths
        
        # 注意: 在评估阶段，我们不使用 Trainer 的 compute_loss，而是使用 generate()
        # 因此 'labels' 字段不是必须的
        return batch_dict

# 推荐指标计算函数
def evaluate_recommendation(
    model: LlamaRecForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    eval_dataset: Dataset,
    eval_collator: EvalDataCollator,
    eval_batch_size: int,
    k_values: List[int] = [10, 20, 50],
    num_beams: int = 50, # Beam search 的束宽
    generation_length: int = 3, # 目标 item 的 tokens 长度
) -> Dict[str, float]:
    model.eval()
    device = model.device
    
    # 构建评估 DataLoader
    eval_dataloader = DataLoader( 
        eval_dataset,
        batch_size=eval_batch_size, 
        collate_fn=eval_collator,
        shuffle=False,
    )
    
    all_ranks = []

    # Tokenizer 的特殊 ID 和词表
    vocab = tokenizer.get_vocab()
    # 假设 <a_id> <b_id> <c_id> 分别对应 'a', 'b', 'c' 开头的 token
    # 你需要知道每个 token codebook 的词表范围
    # 这里我们只关注第一个 codebook 的 token 范围（通常是物品 ID 的主要信息）
    # 示例: 找到所有以 <a_ 开头的 token ID
    first_token_ids = [
        v for k, v in vocab.items() 
        if k.startswith('<a_') and k.endswith('>')
    ]
    
    # LlamaRec 预测目标是下一个物品的 item_token_length 个 tokens
    # 在这个 LlamaRec 的设置中，生成控制是关键
    
    logging.info(f"Starting manual evaluation with num_beams={num_beams}...")
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # 1. 准备输入
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            groundtruth_item_ids = batch['groundtruth_item_ids']
            
            # 2. 生成预测 tokens (Beam Search)
            # 限制生成长度为 item_token_length (例如 3, 对应 <a_id> <b_id> <c_id>)
            # 在 LlamaRec 中，通常会使用自定义的 beam search 约束来确保只生成有效的 item tokens
            # 这里使用标准的 generate 作为示例，但请注意 LlamaRec 可能需要额外的 token 约束
            
            # 由于是 RecSys 场景，通常只需要预测下一个 item 的 tokens
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + generation_length, # 序列长度 + 3 个 tokens
                num_beams=num_beams,
                do_sample=False,
                num_return_sequences=num_beams, # 返回 num_beams 个序列
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # 3. 解码并提取推荐列表
            # generated_ids 的形状是 (batch_size * num_beams, total_length)
            
            # 提取新生成的 tokens (最后 generation_length 个 tokens)
            new_tokens = generated_ids[:, -generation_length:] # (B*K, 3)
            
            # 将生成的 tokens 映射回 Item ID (使用 LlamaRec 的内部机制)
            # 注意: LlamaRec 中 token 到 Item ID 的转换需要依赖 create_semantic_id_tokenizer 的内部逻辑
            # 由于这里没有完整的 LlamaRec 代码，我用一个简化的方式来演示：
            
            # 简易 Item ID 提取：我们只取第一个 token 的 ID (通常是主要 ID)
            predicted_first_tokens = new_tokens[:, 0]
            
            # 将 token ID 转换回 Item ID 字符串
            predicted_item_ids_str = tokenizer.batch_decode(predicted_first_tokens, skip_special_tokens=True)
            
            # 提取出真正的 item id 数字
            # 格式如 "<a_16>"，我们需要提取 "16"
            predicted_item_ids = []
            for token_str in predicted_item_ids_str:
                try:
                    # 匹配 <A_ID> 形式的 token，并提取 ID
                    item_id_str = token_str.strip('<>').split('_')[-1]
                    predicted_item_ids.append(int(item_id_str))
                except Exception:
                    # 如果解码失败，可能是一个特殊 token 或无效 item token
                    predicted_item_ids.append(-1) # 用 -1 标记无效

            # 由于 generate 返回的是 B * K 个序列，我们需要重塑为 (B, K)
            predicted_item_ids_tensor = torch.tensor(predicted_item_ids).view(-1, num_beams).cpu() # (B, K)
            
            # 4. 计算 Hit/Rank
            for i in range(len(groundtruth_item_ids)):
                gt_ids_str = groundtruth_item_ids[i]
                
                # 真正的 ground truth item id 列表 (支持多个)
                gt_ids = set(map(int, gt_ids_str.split(','))) 
                
                # 用户的 Top-K 推荐列表
                user_predictions = predicted_item_ids_tensor[i].tolist()
                
                # 找到第一个命中的 Item 的排名
                # LlamaRec 通常是 Next-Item Prediction，所以只计算一个 Rank
                rank = -1
                
                # 遍历推荐列表，找到第一个命中的排名
                for current_rank, pred_id in enumerate(user_predictions, 1):
                    if pred_id in gt_ids:
                        rank = current_rank
                        break
                
                # 如果命中，记录排名
                if rank != -1:
                    all_ranks.append(rank)

    # 5. 计算最终指标
    if not all_ranks:
        logging.warning("No hits in evaluation. Returning empty metrics.")
        return {}
        
    final_ranks = torch.tensor(all_ranks).float()
    metrics = {}
    
    for k in k_values:
        in_top_k = final_ranks <= k
        hr_k = in_top_k.float().mean().item()
        metrics[f"HR@{k}"] = round(hr_k, 4)
        
        # 计算 NDCG (使用标准公式: log(2)/log(rank+1))
        ndcg_k = (1.0 / torch.log2(final_ranks + 1.0)).where(in_top_k, 0.0).mean().item()
        metrics[f"NDCG@{k}"] = round(ndcg_k, 4)

    metrics["MRR"] = round((1.0 / final_ranks).mean().item(), 4)
    
    return metrics

# 流式指标
# class StreamingMetricsCalculator:   # 这里也用了默认3的设定，看到3要谨慎
#     def __init__(self, k_values: List[int] = [1, 5, 10, 20, 50]):
#         """
#         初始化计算器。

#         Args:
#             k_values (List[int]): 用于计算 HR@k 和 NDCG@k 的 k 值列表。
#         """
#         self.k_values = k_values
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.all_ranks: List[torch.Tensor] = []

#     def __call__(self, eval_preds: EvalPrediction, compute_result: bool) -> Dict[str, float]:
#         logits, labels_matrix = eval_preds.predictions, eval_preds.label_ids

#         num_eval_steps = 3 # 留一法，最后3个token是eval的

#         last_step_logits = logits[0][-num_eval_steps-1:-1, :]
#         labels = labels_matrix.view(-1)[-num_eval_steps:]
        
#         valid_mask = labels != -100
#         labels = labels[valid_mask]
#         last_step_logits = last_step_logits[valid_mask]

#         # 如果这个批次没有有效标签，则直接跳过
#         if labels.numel() > 0:
#             sorted_indices = torch.argsort(last_step_logits, descending=True, dim=-1)
#             ranks = (sorted_indices == labels.unsqueeze(-1)).nonzero(as_tuple=True)[1] + 1
            
#             self.all_ranks.append(ranks.cpu())
#         #print('compute_result: ', compute_result)
#         if compute_result:
#             #print('---------comp')
#             if not self.all_ranks:
#                 return {} # 如果整个评估过程都没有有效标签

#             final_ranks = torch.cat(self.all_ranks).float()
            
#             metrics = {}
#             for k in self.k_values:
#                 in_top_k = final_ranks <= k
#                 hr_k = in_top_k.float().mean().item()
#                 metrics[f"HR@{k}"] = round(hr_k, 4)
                
#                 # 计算 NDCG
#                 ndcg_k = (1.0 / torch.log2(final_ranks + 1.0)).where(in_top_k, 0.0).mean().item()
#                 metrics[f"NDCG@{k}"] = round(ndcg_k, 4)

#             metrics["MRR"] = round((1.0 / final_ranks).mean().item(), 4)
#             print(metrics)
#             self.all_ranks = []
            
#             return metrics
        
#         return {}

# 自定义 Trainer
# 这个 Trainer 可以确保评估时使用我们自定义的 EvalDataCollator
# class CustomTrainer(Trainer):
#     def __init__(self, *args, eval_collator, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.custom_eval_collator = eval_collator

#     def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
#         if eval_dataset is None and self.eval_dataset is None:
#             raise ValueError("Trainer: evaluation requires an eval_dataset.")
#         eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
#         return DataLoader(
#             eval_dataset,
#             batch_size=self.args.eval_batch_size,
#             collate_fn=self.custom_eval_collator,
#             drop_last=self.args.dataloader_drop_last,
#             num_workers=self.args.dataloader_num_workers,
#             pin_memory=self.args.dataloader_pin_memory,
#         )

# Main 函数
def main():
    # 获取配置文件路径
    parser = argparse.ArgumentParser(description="Train a LlamaRec model using a YAML config file.")
    parser.add_argument("--config", type=str, default='KuaiRand-27K_pt', help="Name of the config file to use. For example: pantry")
    cli_args = parser.parse_args()

    # 读取并解析 YAML 配置文件
    logging.info(f"Loading configuration from: {cli_args.config}")
    config_path = os.path.join("pretrain_config", cli_args.config+'.yaml')
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # 从解析的文件中提取配置
    paths_config = config_data['paths']
    model_params = config_data['model_params']
    training_args_dict = config_data['training_args']
    tokenizer_params = config_data['tokenizer_params']
    testing_args = config_data['testing_args']

    # 使用从配置中读取的参数
    dataset_path = paths_config['dataset_path']
    output_dir = paths_config['output_dir']
    tokenizer_dir = paths_config['tokenizer_dir']
    max_seq_length = model_params['max_seq_length']

    # Tokenizer 创建
    tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_file):
        logging.info("Tokenizer not found. Creating a new one ...")
        mock_args = MockTrainingArguments(
            output_dir=tokenizer_dir,
            max_length=max_seq_length,
            codebook_num=tokenizer_params['codebook_num'],
            codeword_num_per_codebook=tokenizer_params['codeword_num_per_codebook'])
        tokenizer = create_semantic_id_tokenizer(mock_args=mock_args)
    else:
        logging.info("Found existing tokenizer. Loading it...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    # 健壮性检查
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if tokenizer.bos_token_id is None: tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    if tokenizer.eos_token_id is None: tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")
    assert tokenizer.pad_token_id is not None and tokenizer.bos_token_id is not None and tokenizer.eos_token_id is not None
    logging.info(f"Final check - pad_token_id: {tokenizer.pad_token_id}, bos_token_id: {tokenizer.bos_token_id}, eos_token_id: {tokenizer.eos_token_id}")

    # 数据集加载
    train_dataset = load_dataset("json", data_files=dataset_path, split='train')
    test_dataset = load_dataset("json", data_files=dataset_path, split='train')

    # 模型构建
    logging.info("Creating LlamaRecForCausalLM model from scratch...")
    config = LlamaRecConfig(
        # 从 model_params 读取架构参数
        hidden_size=model_params['hidden_size'],
        intermediate_size=model_params['intermediate_size'],
        num_hidden_layers=model_params['num_hidden_layers'],
        num_attention_heads=model_params['num_attention_heads'],
        max_position_embeddings=max_seq_length,
        rms_norm_eps=model_params['rms_norm_eps'],
        model_type=model_params['MODEL_TYPE'],
        vocab_size=len(tokenizer),
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaRecForCausalLM(config)
    logging.info(f"Model created with {model.num_parameters() / 1e6:.2f} M parameters.")

    # TrainingArguments 构建
    training_args_dict['output_dir'] = output_dir
    training_args_dict['logging_dir'] = os.path.join(output_dir, 'logs')
    # 使用字典解包来创建 TrainingArguments 实例
    training_args = TrainingArguments(**training_args_dict)

    # DataCollator实例化
    train_collator = TrainDataCollator(tokenizer=tokenizer, max_length=max_seq_length)
    test_collator = EvalDataCollator(tokenizer=tokenizer, max_length=max_seq_length, item_token_length=tokenizer_params['codebook_num'])

    # TODO 测试集 有控制的beamsearch
    # TODO 换框架（不急）
    # Trainer 的实例化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # 我们不将 test_dataset 传入 `eval_dataset`，因为我们要手动运行复杂的生成/评估逻辑
        tokenizer=tokenizer,
        data_collator=train_collator, 
    )

    # 训练和保存
    logging.info("Starting training...")
    trainer.train()

    final_model_path = os.path.join(output_dir, "final")
    logging.info(f"Training complete. Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # # 评估
    # logging.info("Starting custom evaluation...")
    
    # # 调用自定义评估函数
    # metrics = evaluate_recommendation(
    #     model=model,
    #     tokenizer=tokenizer,
    #     eval_dataset=test_dataset,
    #     eval_collator=test_collator,
    #     eval_batch_size=training_args_dict['per_device_eval_batch_size'],
    #     k_values=testing_args['eval_k_values'],
    #     item_token_length=tokenizer_params['codebook_num'],
    #     max_length=max_seq_length,
    #     num_beams=testing_args['num_beams'],
    #     generation_length=tokenizer_params['codebook_num'],
    # )
    
    # logging.info(f"Evaluation Results: {metrics}")

    # logging.info("All operations complete!")

if __name__ == "__main__":
    main()