# train_grpo.py
import sys
import os
import yaml
import argparse
from datetime import datetime
import torch
from datasets import load_dataset
from tqdm import tqdm
from trl import GRPOConfig
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    LogitsProcessorList, 
    TrainerCallback, 
    EarlyStoppingCallback,
)
from torch.utils.data import DataLoader
import numpy as np

# ================= 配置路径 =================
# 1. 确保能导入相关模块
sys.path.append("../Rec-Transformer") # LlamaRec 所在
sys.path.append("../")

# 导入自定义模块
from CTR_models.DIN.DIN_evaluator import DINScorer
from llamarec import LlamaRecConfig, LlamaRecForCausalLM 
from utils.RL_utils import GRPOTrainer_not_skip_special_token, DINRewardRunner
from utils.utils_evaluate import DynamicHierarchicalLogitsProcessor, build_item_token_codebooks_dynamically
from utils.eval import compute_hr_at_k, compute_ndcg_at_k

# ================= 注册模型 =================
AutoConfig.register("llama-rec", LlamaRecConfig)
AutoModelForCausalLM.register(LlamaRecConfig, LlamaRecForCausalLM)


class GRPO_Eval_Trainer(GRPOTrainer_not_skip_special_token):
    def __init__(self, eval_dataset, generation_config_params, **kwargs):
        """
        继承自 GRPOTrainer，但注入了 CustomTrainer 的评测逻辑
        """
        self.custom_eval_dataset = eval_dataset
        # 解包生成参数
        self.gen_len = generation_config_params.get('generation_length', 4)
        self.num_beams = generation_config_params.get('num_beams', 1)
        self.k_values = generation_config_params.get('k_values', [1, 5, 10])
        self.item_token_codebooks = generation_config_params.get('item_token_codebooks', None)

        # --- 【核心移植】构建 NumPy 向量化查找表 ---
        # 这部分逻辑直接来自你的 CustomTrainer
        print(">>> Building NumPy Vectorized Vocab Table for Evaluation...")
        vocab = kwargs['processing_class'].get_vocab()
        max_id = max(vocab.values())
        # 初始化 object 数组
        self.vocab_array = np.array(["" for _ in range(max_id + 1)], dtype=object)
        for k, v in vocab.items():
            self.vocab_array[v] = k
        print(">>> ✅ Vocab Table built.")

        # 3. 【关键修改】调用父类初始化时，显式传入 eval_dataset
        super().__init__(eval_dataset=eval_dataset, **kwargs)

    # 简单的 Collator：专门用于评估时的 tokenization
    # 因为 RL Dataset 是 {'prompt': str, 'ground_truth': str}，需要转 Tensor
    def _eval_collator(self, batch):
        prompts = [x['prompt'] for x in batch]
        ground_truths = [x['ground_truth'] for x in batch]
        
        # 实时 Tokenize
        inputs = self.processing_class(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            padding_side='left',
            max_length=200,
        )
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "groundtruth": ground_truths
        }

    # 重写 evaluate 方法
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # 使用我们传入的 dataset
        eval_ds = eval_dataset if eval_dataset is not None else self.custom_eval_dataset
        
        if eval_ds is None:
            print(">>> Warning: No eval dataset provided, skipping evaluation.")
            return {}

        # 1. 准备 DataLoader
        # 使用 args.per_device_eval_batch_size
        batch_size = self.args.per_device_eval_batch_size or self.args.per_device_train_batch_size
        eval_dataloader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            collate_fn=self._eval_collator,
            shuffle=False,
            drop_last=False
        )

        # 切换模式
        model = self.model
        model.eval()
        
        print(f"\n***** Running Generative Evaluation (Step {self.state.global_step}) *****")
        
        total_metrics_sum = {f"HR@{k}": 0.0 for k in self.k_values}
        total_metrics_sum.update({f"NDCG@{k}": 0.0 for k in self.k_values})
        total_samples = 0
        
        # 2. 循环生成
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.args.device)
                attention_mask = batch['attention_mask'].to(self.args.device)
                groundtruth = batch['groundtruth'] # List[str]

                curr_bs = input_ids.shape[0]
                prompt_length = input_ids.shape[1]

                # 构造 Logits Processor (如果提供了 codebooks)
                logits_processor = LogitsProcessorList()
                if self.item_token_codebooks:
                    logits_processor.append(
                        DynamicHierarchicalLogitsProcessor(
                            prompt_length=prompt_length,
                            item_token_codebooks=self.item_token_codebooks,
                            device=self.args.device
                        )
                    )

                # 3. Beam Search 生成 (HuggingFace Generate)
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=prompt_length + self.gen_len,
                    num_beams=self.num_beams,
                    do_sample=False, 
                    num_return_sequences=self.num_beams,
                    pad_token_id=self.processing_class.pad_token_id,
                    eos_token_id=self.processing_class.eos_token_id,
                    logits_processor=logits_processor,
                    use_cache=True
                )

                # 4. 【极速解码】(NumPy Vectorized)
                # shape: [Batch_Size * Num_Beams, Gen_Len]
                new_tokens_cpu = generated_ids[:, -self.gen_len:].cpu().numpy()
                token_strs = self.vocab_array[new_tokens_cpu] # O(1) 查表
                
                # 向量化字符串拼接
                if self.gen_len == 1:
                    predicted_token_sequences = token_strs.flatten().tolist()
                else:
                    result_array = token_strs[:, 0]
                    for i in range(1, self.gen_len):
                        result_array = result_array + token_strs[:, i]
                    predicted_token_sequences = result_array.tolist()

                # 5. Reshape 为 [Batch, Num_Beams]
                reshaped_token_sequences = [
                    predicted_token_sequences[i : i + self.num_beams]
                    for i in range(0, len(predicted_token_sequences), self.num_beams)
                ]

                # 6. 计算指标
                batch_hr = compute_hr_at_k(reshaped_token_sequences, groundtruth, self.k_values)
                batch_ndcg = compute_ndcg_at_k(reshaped_token_sequences, groundtruth, self.k_values)

                for k_val in self.k_values:
                    total_metrics_sum[f"HR@{k_val}"] += batch_hr[f"HR@{k_val}"] * curr_bs
                    total_metrics_sum[f"NDCG@{k_val}"] += batch_ndcg[f"NDCG@{k_val}"] * curr_bs
                
                total_samples += curr_bs
        
        # 恢复训练模式
        model.train()

        # 7. 汇总并 Log
        metrics = {f"{metric_key_prefix}_{k}": (v / total_samples) for k, v in total_metrics_sum.items()}
        
        # 调用 Trainer 内置的 log 方法，这样 Wandb 和日志文件都能记录到
        self.log(metrics)
        
        # 这一步很关键：将 metric 传回给 Trainer 的 control 系统，用于早停判断
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        
        print(f">>> Evaluation Metrics: {metrics}")
        return metrics


if __name__ == '__main__':
    # ================= 命令行参数解析 =================
    parser = argparse.ArgumentParser(description="GRPO Training with YAML Config")
    parser.add_argument("--config", type=str, default="grpo_config.yaml", help="Path to the YAML config file")
    args_cli = parser.parse_args()

    # ================= 加载 config =================
    print(f">>> Loading configuration from {args_cli.config}...")
    with open(args_cli.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 提取各个部分的配置
    paths_cfg = config['paths']
    din_cfg = config['din']
    train_cfg = config['training']
    eval_cfg = config['evaluation']

    # ================= 加载 DIN Scorer =================
    print(">>> Initializing DIN Reward Model...")
    
    din_scorer = DINScorer(
        config_dir=paths_cfg['din_config_dir'],
        model_dir=paths_cfg['din_model_dir'],
        experiment_id=din_cfg['experiment_id'],
        data_dir=paths_cfg['din_data_dir'],
        device=din_cfg['device']                   
    )
    print(">>> DIN Scorer Ready.")

    reward_runner = DINRewardRunner(
        scorer=din_scorer, 
        weight=din_cfg['reward_weight'],
        penalty=din_cfg.get('penalty', -1.0)
    )

    # ================= 数据集加载 =================
    rl_data_dir = paths_cfg['rl_data_dir']
    
    # 1. 加载 Train
    train_json_path = os.path.join(rl_data_dir, paths_cfg['train_file'])
    print(f">>> Loading Train Dataset: {train_json_path}")
    train_dataset = load_dataset("json", data_files=train_json_path, split='train')

    # 2. 加载 Valid
    test_json_path = os.path.join(rl_data_dir, paths_cfg['test_file'])
    print(f">>> Loading Test Dataset: {test_json_path}")
    test_dataset = load_dataset("json", data_files=test_json_path, split='train')

    # ================= 训练输出路径配置 =================
    base_dir = paths_cfg['output_root']
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"checkpoints_{date_time}")
    os.makedirs(output_dir, exist_ok=True)
    print(f">>> Checkpoints will be saved to: {output_dir}")

    # 逻辑：总日志目录 / 本次实验名称(带时间戳)
    # 这样你在 tensorboard --logdir ./temp_try_GRPO_Rec_Output/all_tensorboard_logs 时能看到所有实验的曲线对比
    tb_root = paths_cfg.get('tensorboard_root', './temp_try_GRPO_Rec_Output/all_tensorboard_logs')
    tb_dir = os.path.join(tb_root, f"run_{date_time}")
    
    print(f">>> TensorBoard logs will be saved to: {tb_dir}")

    # ================= 训练参数配置 =================
    # 从 yaml 中提取 evaluate/save 的步数
    eval_save_steps = train_cfg['eval_save_steps']

    training_args = GRPOConfig(
        output_dir=output_dir,
        logging_dir = tb_dir,
        report_to="tensorboard",
        
        learning_rate=float(train_cfg['learning_rate']), # 确保 YAML 读取的是 float
        num_train_epochs=train_cfg['num_train_epochs'],
        per_device_train_batch_size=train_cfg['per_device_train_batch_size'],
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
        
        # 日志
        logging_steps=train_cfg['logging_steps'],
        
        # 生成参数 (RL Training)
        max_completion_length=train_cfg['max_completion_length'],
        num_generations=train_cfg['num_generations'],
        use_vllm=train_cfg['use_vllm'],
        bf16=train_cfg['bf16'],

        # 评估与早停策略
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        per_device_eval_batch_size=train_cfg['per_device_train_batch_size'], # 默认和 train 一样

        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=train_cfg['save_total_limit'],
        
        load_best_model_at_end=True,
        metric_for_best_model=train_cfg['metric_for_best_model'],
        greater_is_better=True,
        
        # 防止删掉 prompt/ground_truth 列
        remove_unused_columns=False
    )

    # ================= 加载模型与Tokenizer =================
    print(">>> Loading LLM...")
    llm_path = paths_cfg['llm_model_path']
    model = LlamaRecForCausalLM.from_pretrained(llm_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_path)

    # Tokenizer 补丁
    if tokenizer.model_input_names is not None and "token_type_ids" in tokenizer.model_input_names:
        tokenizer.model_input_names.remove("token_type_ids")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ================= 准备评估用的 Codebook =================
    generation_length = eval_cfg['generation_length']
    item_token_codebooks = build_item_token_codebooks_dynamically(tokenizer, generation_length)
    
    # 组装评估配置字典
    eval_config_dict = {
        "generation_length": generation_length,
        "num_beams": eval_cfg['num_beams'],
        "k_values": eval_cfg['k_values'],
        "item_token_codebooks": item_token_codebooks
    }

    # ================= 初始化 Trainer =================
    trainer = GRPO_Eval_Trainer(
        model=model,
        reward_funcs=[reward_runner],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        generation_config_params=eval_config_dict,
        processing_class=tokenizer,
        
        # 早停回调
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=train_cfg['early_stopping_patience']
        )]
    )

    # ================= 开始训练 =================
    print(">>> Starting GRPO Training with Live Evaluation...")
    trainer.train()

    # ================= 打印最佳模型结果 =================
    # 获取最佳 Checkpoint 的路径
    best_ckpt_path = trainer.state.best_model_checkpoint
    
    if best_ckpt_path:
        print(f"\n" + "="*50)
        print(f"🏆 TRAINING FINISHED. BEST MODEL FOUND.")
        print(f"="*50)
        print(f"📍 Best Checkpoint Path: {best_ckpt_path}")
        print(f"🌟 Best Metric Value:    {trainer.state.best_metric}")
        
        # --- 核心逻辑：从日志历史中捞出最佳那一步的完整指标 ---
        # 1. 从路径中提取最佳步数 (例如 "xxx/checkpoint-500" -> 500)
        try:
            best_step = int(best_ckpt_path.split('-')[-1])
            
            # 2. 遍历日志历史找到那一刻的详细数据
            best_log_entry = None
            for log in trainer.state.log_history:
                # 必须同时满足：是这一步，且包含评估指标(比如有 eval_loss 或 eval_NDCG@10)
                if log.get("step") == best_step and "eval_NDCG@10" in log:
                    best_log_entry = log
                    break
            
            if best_log_entry:
                print(f"\n📊 Detailed Metrics for Best Model (Step {best_step}):")
                # 格式化打印字典
                for k, v in best_log_entry.items():
                    if k.startswith("eval_"):
                        print(f"   - {k}: {v}")
            else:
                print(f"⚠️ Could not find detailed logs for step {best_step} in history.")

        except Exception as e:
            print(f"⚠️ Error parsing best step info: {e}")

    # ================= 保存最终模型 =================
    final_save_path = os.path.join(output_dir, "final_best_grpo_model")
    trainer.save_model(final_save_path)
    print(f">>> Training Finished & Best Model Saved to {final_save_path}")