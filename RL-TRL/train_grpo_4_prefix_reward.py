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
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)
from torch.utils.data import DataLoader
import numpy as np
import random
import re
from collections import defaultdict

# ================= 配置路径 =================
# 1. 确保能导入相关模块
sys.path.append("../Rec-Transformer") # LlamaRec 所在
sys.path.append("../")


# 导入自定义模块
from CTR_models.src.DIN_evaluator import DINScorer
from llamarec import LlamaRecConfig, LlamaRecForCausalLM 
from sasrec import SasRecForCausalLM
from utils.RL_utils import GRPOTrainer_not_skip_special_token, RewardRunner
from utils.datacollator import EvalDataCollator, preprocess_function
from utils.utils_evaluate import DynamicHierarchicalLogitsProcessor, build_item_token_codebooks_dynamically
from utils.eval import compute_hr_at_k, compute_ndcg_at_k
from utils.utils import *

# ================= 注册模型 =================
AutoConfig.register("llama-rec", LlamaRecConfig)
AutoModelForCausalLM.register(LlamaRecConfig, LlamaRecForCausalLM)

class GRPO_Eval_Trainer(GRPOTrainer_not_skip_special_token):
    def __init__(self, eval_dataset, generation_config_params, eval_collator, **kwargs):
        """
        继承自 GRPOTrainer，但注入了与 Script B 完全对齐的评测逻辑
        """
        self.custom_eval_dataset = eval_dataset
        self.eval_collator = eval_collator  # 接收来自 Script B 的 Collator
        
        # 解包生成参数
        self.gen_len = generation_config_params.get('generation_length', 4)
        self.num_beams = generation_config_params.get('num_beams', 1)
        self.k_values = generation_config_params.get('k_values', [1, 5, 10])
        self.item_token_codebooks = generation_config_params.get('item_token_codebooks', None)
        self.eval_sample_num = generation_config_params.get('eval_sample_num', 2000)

        # --- 构建 NumPy 向量化查找表 ---
        print(">>> Building NumPy Vectorized Vocab Table for Evaluation...")
        vocab = kwargs['processing_class'].get_vocab()
        max_id = max(vocab.values())
        self.vocab_array = np.array(["" for _ in range(max_id + 1)], dtype=object)
        for k, v in vocab.items():
            self.vocab_array[v] = k
        print(">>> ✅ Vocab Table built.")

        # 调用父类初始化
        # 注意：父类不需要 eval_dataset，因为我们要在 evaluate 中手动处理它
        # 避免父类对我们的 eval_dataset 做不必要的列移除操作
        super().__init__(eval_dataset=eval_dataset, **kwargs)

    # 重写 evaluate 方法
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # 优先使用传入的 dataset，否则使用初始化时预处理好的 dataset
        eval_ds = eval_dataset if eval_dataset is not None else self.custom_eval_dataset
        
        if eval_ds is None:
            print(">>> Warning: No eval dataset provided, skipping evaluation.")
            return {}
        
        if metric_key_prefix == "eval" and eval_ds is not None:
            total_size = len(eval_ds)
            if total_size > self.eval_sample_num:
                print(f"⚡ [SpeedUp] Sampling {self.eval_sample_num} random examples from {total_size} for validation.")
                
                # 随机选取索引
                # 注意：这里每次验证都会重新随机，导致验证指标会有波动，但能更全面地监控模型
                random_indices = random.sample(range(total_size), self.eval_sample_num)
                
                # 使用 HuggingFace dataset 的 select 方法创建子集
                eval_ds = eval_ds.select(random_indices)
            else:
                print(f"Dataset size ({total_size}) <= {self.eval_sample_num}, running full evaluation.")

        # 1. 准备 DataLoader
        # 确保数据经过了 preprocess_function 处理
        batch_size = self.args.per_device_eval_batch_size or self.args.per_device_train_batch_size
        eval_dataloader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            collate_fn=self.eval_collator, # 使用传入的正确 Collator
            shuffle=False,
            drop_last=False
        )

        # 切换模式
        model = self.model
        model.eval()
        
        print(f"\n***** Running Generative Evaluation (Step {self.state.global_step}) *****")
        print(f"  Num examples = {len(eval_ds)}")
        print(f"  Batch size = {batch_size}")
        
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
        
        # 从打印结果来看，好像transformers会自动打印metric，和下面的几乎别无二致
        # print(f">>> Evaluation Metrics: {metrics}")
        return metrics


class RewardRunner_prefix:
    # ================= 内部类定义 =================
    class TrieNode:
        def __init__(self):
            # 使用字典存储子节点，key为token字符串
            self.children = {}
            # 经过该节点的路径数量（流行度）
            self.count = 0
            
    # ================= 主类逻辑 =================
    def __init__(self, scorer=None, weight=0.8, trie_weight=1.0, penalty=-1.0, name="reward_combined"):
        """
        :param scorer: DINScorer 实例 (可选)
        :param weight: DIN 分数权重
        :param trie_weight: 前缀树 Token-level 奖励的权重
        :param penalty: 格式错误惩罚
        """
        self.scorer = scorer
        self.weight = weight
        self.trie_weight = trie_weight
        self.penalty = penalty
        self.__name__ = name

    def _parse_items(self, text):
        """正则提取 <item_id>"""
        return re.findall(r"<[^>]+>", text)

    def _build_batch_trie(self, ground_truths):
        """
        为单个样本构建 Trie 树。
        :param ground_truths: List[str] 或 str。当前 User 的所有正确答案路径。
        """
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
            
        root = self.TrieNode()
        # 根节点的 count 等于该样本所有 GT 的总数
        root.count = len(ground_truths)
        
        for gt_str in ground_truths:
            tokens = self._parse_items(gt_str)
            node = root
            for token in tokens:
                if token not in node.children:
                    node.children[token] = self.TrieNode()
                node = node.children[token]
                # 路径经过此节点，计数+1
                node.count += 1
        return root

    def _compute_token_level_trie_score(self, completion, ground_truth):
        """
        计算累加的 Token-level 概率奖励。
        """
        # 1. 格式校验
        c_stripped = completion.strip()
        if not c_stripped.startswith("<") or not c_stripped.endswith(">"):
            return self.penalty

        # 2. 解析生成的序列
        gen_tokens = self._parse_items(c_stripped)
        if not gen_tokens:
            return self.penalty

        # 3. 构建 Trie (针对当前样本的 GT 集合)
        root = self._build_batch_trie(ground_truth)

        # 4. 逐 Token 匹配并计算概率
        current_node = root
        accumulated_prob_score = 0.0
        
        for token in gen_tokens:
            if token in current_node.children:
                next_node = current_node.children[token]
                
                # === 核心算法：Token 粒度的概率 ===
                # 父节点有 N 条路，其中 M 条路走了当前 token
                # Reward_t = M / N
                # 这意味着模型走了一条“大路”（高流行度路径）会得高分，走“小路”得低分
                step_reward = next_node.count / current_node.count
                
                accumulated_prob_score += step_reward
                
                # 指针下移
                current_node = next_node
            else:
                # 匹配中断：后续 Token 无法在 GT 树中找到，停止奖励
                # 这里可以选择给一个小的 step penalty，或者直接 break
                break
        
        if len(gen_tokens) > 0:
            return accumulated_prob_score / len(gen_tokens)
        else:
            return 0.0

    def __call__(self, prompts, completions, ground_truth, user_id, **kwargs):
        """
        TRL 回调入口
        """
        # 这里的 prompts 其实就是 history，根据你的代码逻辑
        history = prompts
        
        # 结果列表
        final_rewards = [0.0] * len(completions)
        
        # 收集需要 DIN 打分的样本
        din_batch_indices = []
        din_batch_uids = []
        din_batch_hist = []
        din_batch_comp = []

        for i, (c, gt) in enumerate(zip(completions, ground_truth)):
            
            if c in gt:
                # 直接命中 GT，给最高分
                final_rewards[i] = 1.0
                continue
            # --- 部分 1: 前缀树概率奖励 (Token-level Accumulation) ---
            # 这是一个密集奖励 (Dense Reward)
            trie_score = self._compute_token_level_trie_score(c, gt)
            
            # 如果格式错误，直接惩罚并跳过 DIN
            if trie_score == self.penalty:
                final_rewards[i] = self.penalty
                continue
            
            final_rewards[i] = trie_score * self.trie_weight

            # --- 部分 2: 准备 DIN 打分 (Sequence-level Reward) ---
            # 只有当模型配置了 scorer 且权重不为0时才计算
            if self.scorer and self.weight > 0:
                din_batch_indices.append(i)
                din_batch_uids.append(user_id[i])
                din_batch_hist.append(history[i])
                din_batch_comp.append(c)

        # --- 部分 3: 批量执行 DIN ---
        if din_batch_indices:
            try:
                # 假设 scorer.predict_batch 返回 list of floats
                din_scores = self.scorer.predict_batch(
                    user_ids=din_batch_uids,
                    history=din_batch_hist,
                    completions=din_batch_comp
                )
                
                for idx, d_score in zip(din_batch_indices, din_scores):
                    # 叠加奖励： Token累加分 + DIN整句分
                    final_rewards[idx] += max(-0.1, d_score) * self.weight
                    
            except Exception as e:
                # 容错处理，避免训练中断，仅打印错误
                print(f"[RewardRunner Error] DIN inference failed: {e}")

        return final_rewards

if __name__ == '__main__':
    # ================= 命令行参数解析 =================
    parser = argparse.ArgumentParser(description="GRPO Training with YAML Config")
    parser.add_argument("--config", type=str, default="./rl_configs/KuaiRec_big_llamarec_DIN.yaml", help="Path to the YAML config file")
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

    reward_runner = RewardRunner_prefix(
        scorer=din_scorer, 
        weight=din_cfg['reward_weight'],
        penalty=din_cfg.get('penalty', -1.0)
    )

    # ================= 加载模型与Tokenizer =================
    print(">>> Loading LLM...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_path = paths_cfg['llm_model_path']
    # 动态选择模型类
    if train_cfg['model_name'] == 'sasrec':
        model_class = SasRecForCausalLM
    elif train_cfg['model_name'] == 'llama':
        model_class = LlamaForCausalLM
    elif train_cfg['model_name'].startswith('qwen'):
        model_class = Qwen2ForCausalLM
    else:
        model_class = LlamaRecForCausalLM

    try:
        model = model_class.from_pretrained(
            llm_path, 
            torch_dtype=torch.bfloat16 if train_cfg.get('bf16', False) else "auto",
            device_map=device
        )
    except Exception as e:
        print(f"Failed to load model from {llm_path}. Error: {e}")
        # 尝试自动回退
        print("Trying AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(llm_path, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)

    # Tokenizer 补丁
    if tokenizer.model_input_names is not None and "token_type_ids" in tokenizer.model_input_names:
        tokenizer.model_input_names.remove("token_type_ids")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    # ================= 准备评估用的 Codebook =================
    generation_length = eval_cfg['generation_length']
    item_token_codebooks = build_item_token_codebooks_dynamically(tokenizer, generation_length)

    # ================= 数据集加载 =================
    rl_data_dir = paths_cfg['rl_data_dir']
    
    # 1. 加载 Train
    train_json_path = os.path.join(rl_data_dir, paths_cfg['train_file'])
    print(f">>> Loading Train Dataset: {train_json_path}")
    train_dataset = load_dataset("json", data_files=train_json_path, split="train")

    # 2. 加载 Test
    test_json_path = os.path.join(rl_data_dir, paths_cfg['test_file'])
    print(f">>> Loading Test Dataset: {test_json_path}")
    raw_test_dataset = load_dataset("json", data_files=test_json_path, split="train")

    # ================= 训练输出路径配置 =================
    base_dir = paths_cfg['output_root']
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"checkpoints_{date_time}")
    os.makedirs(output_dir, exist_ok=True)
    print(f">>> Checkpoints will be saved to: {output_dir}")

    max_seq_length = train_cfg['max_seq_length']
    print(f">>> Preprocessing Test Dataset... Max Length: {max_seq_length}")
    # 使用 Script B 同款的 preprocess_function
    test_dataset = raw_test_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=8, 
        remove_columns=['prompt'], # 这里要注意：preprocess_function 会生成 input_ids，我们只保留需要的列
        fn_kwargs={"tokenizer": tokenizer, "max_seq_length": max_seq_length},
        desc="Tokenizing Test Set"
    )
    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=max_seq_length)

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
        bf16=train_cfg['bf16'] if 'bf16' in train_cfg else False,
        mask_truncated_completions=train_cfg.get('mask_truncated_completions', False),
        temperature=0.7,        # 稍微降一点，避免生成完全乱码的 Item ID
        top_k=50,               # 限制采样范围，避免采样到极其冷门的 Item
        top_p=0.95,

        # 评估与早停策略
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        per_device_eval_batch_size=train_cfg['per_device_eval_batch_size'],

        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=train_cfg['save_total_limit'],
        
        load_best_model_at_end=True,
        metric_for_best_model=train_cfg['metric_for_best_model'],
        greater_is_better=True,
        
        # 防止删掉 prompt/ground_truth 列
        remove_unused_columns=False
    )
    
    # 组装评估配置字典
    eval_config_dict = {
        "generation_length": generation_length,
        "num_beams": eval_cfg['num_beams'],
        "k_values": eval_cfg['k_values'],
        "item_token_codebooks": item_token_codebooks,
        "eval_sample_num": eval_cfg.get('eval_sample_num', 2000)
    }

    # ================= 初始化 Trainer =================
    trainer = GRPO_Eval_Trainer(
        model=model,
        reward_funcs=[reward_runner],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        eval_collator=eval_collator,
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