import os
import json
import logging
from datasets import load_dataset
import yaml
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from typing import List
from transformers import EarlyStoppingCallback
import sys
import transformers
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
from util.datacollator import TrainDataCollator, EvalDataCollator
from util.utils_evaluate import build_item_token_codebooks_dynamically, beamsearch_prefix_constraint_fn
from util.eval import compute_hr_at_k, compute_ndcg_at_k
import warnings
# 忽略特定的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.trainer")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class CustomTrainer(Trainer):
    def __init__(self, eval_collator, generation_config_params, **kwargs):
        super().__init__(**kwargs)
        self.eval_collator = eval_collator
        # 将生成需要的参数存下来
        self.gen_len = generation_config_params['generation_length']
        self.num_beams = generation_config_params['num_beams']
        self.k_values = generation_config_params['k_values']
        self.item_token_codebooks = generation_config_params['item_token_codebooks']

    # 重写 evaluate 方法
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # 使用传入的 eval_dataset 或者初始化时传入的 dataset
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        
        # 1. 准备 DataLoader
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.eval_collator, # 使用评估专用的 Collator
            shuffle=False,
            drop_last=False
        )

        # 2. 准备模型和设备
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()
        
        logging.info(f"***** Running Custom Evaluation (Generation) *****")
        
        total_metrics_sum = {f"HR@{k}": 0.0 for k in self.k_values}
        total_metrics_sum.update({f"NDCG@{k}": 0.0 for k in self.k_values})
        total_samples = 0

        # 3. 循环生成
        with torch.no_grad():
            for batch in eval_dataloader:
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.args.device)
                attention_mask = batch['attention_mask'].to(self.args.device)
                groundtruth = batch['groundtruth'] # groundtruth 不需要上 GPU

                batch_size = input_ids.shape[0]
                prompt_length = input_ids.shape[1]

                # 定义约束闭包
                def batch_beamsearch_prefix_constraint_fn(batch_id: int, input_ids_tensor: torch.Tensor) -> List[int]:
                    return beamsearch_prefix_constraint_fn(
                        batch_id=batch_id,
                        input_ids_tensor=input_ids_tensor,
                        prompt_length=prompt_length,
                        generation_length=self.gen_len,
                        item_token_codebooks=self.item_token_codebooks
                    )

                # 生成
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=prompt_length + self.gen_len,
                    num_beams=self.num_beams,
                    do_sample=False,
                    num_return_sequences=self.num_beams,
                    pad_token_id=self.processing_class.pad_token_id,
                    eos_token_id=self.processing_class.eos_token_id,
                    prefix_allowed_tokens_fn=batch_beamsearch_prefix_constraint_fn,
                    use_cache=True
                )

                # 解码
                new_tokens = generated_ids[:, -self.gen_len:]
                predicted_token_sequences = self.processing_class.batch_decode(new_tokens, skip_special_tokens=False)
                # Reshape: [batch_size, num_beams]
                reshaped_token_sequences = [
                    predicted_token_sequences[i : i + self.num_beams]
                    for i in range(0, len(predicted_token_sequences), self.num_beams)
                ]

                # 计算指标
                batch_hr = compute_hr_at_k(reshaped_token_sequences, groundtruth, self.k_values)
                batch_ndcg = compute_ndcg_at_k(reshaped_token_sequences, groundtruth, self.k_values)

                for k_val in self.k_values:
                    total_metrics_sum[f"HR@{k_val}"] += batch_hr[f"HR@{k_val}"] * batch_size
                    total_metrics_sum[f"NDCG@{k_val}"] += batch_ndcg[f"NDCG@{k_val}"] * batch_size
                
                total_samples += batch_size

        # 4. 汇总指标
        metrics = {f"{metric_key_prefix}_{k}": (v / total_samples) for k, v in total_metrics_sum.items()}
        
        # 5. 记录日志 (关键：这样 Trainer 才能看到这些指标并用于早停)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        
        logging.info(f"Evaluation metrics: {metrics}")
        return metrics

# Main 函数
def main():
    # 获取配置文件路径
    parser = argparse.ArgumentParser(description="Train a LlamaRec model using a YAML config file.")
    parser.add_argument("--dataset", type=str, default='Beauty')
    parser.add_argument("--model_name", type=str, default='llama-rec')
    args = parser.parse_args()

    # 读取并解析 YAML 配置文件
    config_path = os.path.join("pretrain_config", args.dataset+'_'+ args.model_name + '.yaml')
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # 从解析的文件中提取配置
    paths_config = config_data['paths']
    model_params = config_data['model_params']
    training_args_dict = config_data['training_args']
    tokenizer_params = config_data['tokenizer_params']
    testing_args = config_data['testing_args']

    # 使用从配置中读取的参数
    dataset_path = os.path.join(paths_config['dataset_path'], 'train_data.json')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(paths_config['output_dir'], f"{model_params['MODEL_TYPE']}_{timestamp}")
    tokenizer_dir = paths_config['tokenizer_dir']
    max_seq_length = model_params['max_seq_length']
    generation_length = tokenizer_params['codebook_num']

    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # ================= 日志配置区域开始 =================
    log_file_path = os.path.join(output_dir, "training_process.log")
    
    # 1. 创建一个文件处理器 (File Handler)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    file_handler.setFormatter(formatter)
    
    # 2. 创建一个屏幕流处理器 (Stream Handler) - 让你在终端也能看到
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # 3. 【关键】强制配置 根日志 (Root Logger) - 负责你写的 logging.info
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # 先清空可能存在的旧 handler，防止重复打印
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # 4. 配置 Transformers 的日志 - 负责 Trainer 的内部输出
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # 拿到 transformers 的专用 logger 并把文件处理器挂上去
    transformers_logger = transformers.utils.logging.get_logger("transformers")
    transformers_logger.addHandler(file_handler)
    
    logging.info(f"✅ Logging started. Output file: {log_file_path}")
    formatted_config = json.dumps(config_data, indent=4, ensure_ascii=False)
    logging.info(f"🚀 Loaded Configuration:\n{formatted_config}")
    # =================  日志配置区域结束 =================


    # Tokenizer 创建
    tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_file):
        logging.info("Tokenizer not found. Creating a new one ...")
        mock_args = MockTrainingArguments(
            output_dir=tokenizer_dir,
            max_length=max_seq_length + generation_length,
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
    eval_dataset = load_dataset("json", data_files=dataset_path, split='train')

    # 模型构建
    logging.info("Creating model from scratch...")
    config = LlamaRecConfig(
        # 从 model_params 读取架构参数
        hidden_size=model_params['hidden_size'],
        intermediate_size=model_params['intermediate_size'],
        num_hidden_layers=model_params['num_hidden_layers'],
        num_attention_heads=model_params['num_attention_heads'],
        max_position_embeddings=max_seq_length + generation_length,
        rms_norm_eps=model_params['rms_norm_eps'],
        model_type=model_params['MODEL_TYPE'],
        vocab_size=len(tokenizer),
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_levels=tokenizer_params['codebook_num'],
        # history_weights=model_params['history_weights'],
        # enable_hierarchical_prediction=model_params['enable_hierarchical_prediction'],
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
    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=max_seq_length)

    item_token_codebooks = build_item_token_codebooks_dynamically(tokenizer, generation_length)
    generation_config_params = {
        "generation_length": generation_length,
        "num_beams": testing_args['num_beams'],
        "k_values": testing_args['eval_k_values'],
        "item_token_codebooks": item_token_codebooks
    }

    # Trainer 的实例化
    # 使用自定义的 Trainer，并传入 eval_dataset 和回调
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,   # 传入验证集
        processing_class=tokenizer,
        data_collator=train_collator,
        eval_collator=eval_collator,
        generation_config_params=generation_config_params,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)] 
    )

    # 训练和保存
    logging.info("Starting training...")
    trainer.train()

    # 打印历史最优指标
    if trainer.state.best_model_checkpoint:
        best_metric_name = training_args.metric_for_best_model
        best_metric_val = trainer.state.best_metric
        logging.info("=" * 40)
        logging.info(f"🏆 训练结束，历史最优结果如下：")
        logging.info(f"最优模型路径: {trainer.state.best_model_checkpoint}")
        # 如果你配置了 metric_for_best_model，这里会显示具体数值
        logging.info(f"最优指标 ({best_metric_name}): {best_metric_val}")
        logging.info("=" * 40)
    else:
        logging.info("⚠️ 未找到最优模型记录 (请检查 YAML 中是否设置了 load_best_model_at_end=True)")

    # 保存模型
    # 注意：如果 load_best_model_at_end=True，trainer.train() 结束时模型参数已经是“最优的”了
    final_model_path = os.path.join(output_dir, "best_model")
    logging.info(f"Saving model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logging.info("All operations complete!")

if __name__ == "__main__":
    main()