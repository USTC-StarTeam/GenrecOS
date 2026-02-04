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
from trl.trainer.utils import (
    nanmax,
    nanmin,
)
from torch.utils.data import DataLoader
import numpy as np
import random

# ================= 配置路径 =================
# 1. 确保能导入相关模块
sys.path.append("../Rec-Transformer") # LlamaRec 所在
sys.path.append("../")


# 导入自定义模块
from CTR_models.src.DIN_evaluator import DINScorer
from llamarec import LlamaRecConfig, LlamaRecForCausalLM 
from sasrec import SasRecForCausalLM
from utils.RL_utils import GRPOTrainer_not_skip_special_token, RewardRunner, RewardRunner_wo_gt
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
        # self.token_pos_weights = torch.tensor([1.0, 0.5, 0.3, 0.2], dtype=torch.float32)
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

class GRPO_Eval_Trainer_Confidence_Aware(GRPO_Eval_Trainer):
   # 主要传入token_pos_weights
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        # In the base GRPO implementation, advantages are expected to have shape (B,). To support subclasses that
        # provide advantages with shape (B, T) (e.g., MiniLLM), we *conditionally* unsqueeze the tensor.
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            # Importance sampling correction for the KL divergence
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            per_token_loss = torch.empty_like(coef_1)
            positive_advantages_mask = advantages.repeat([1, coef_1.shape[1]]) > 0
            per_token_loss[positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[positive_advantages_mask], self.args.sapo_temperature_pos
            )
            per_token_loss[~positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[~positive_advantages_mask], self.args.sapo_temperature_neg
            )
            per_token_loss = -per_token_loss * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]

        # 创建一个专门用于计算 Loss 的 mask，避免污染用于记录日志的原始 mask
        loss_mask = mask.clone().float()

        # ================== 🔵 新增：动态置信度加权 (Confidence-Aware Weighting) 🔵 ==================
        # 目的：对于 Advantage > 0 (好样本)，关注置信度低的 (难样本)
        #       对于 Advantage < 0 (坏样本)，关注置信度高的 (傲慢样本)
        
        # 1. 获取当前 Token 的生成概率 P (切断梯度，只作为权重系数)
        token_probs = torch.exp(per_token_logps.detach())
        
        # 2. 扩展 Advantage 维度以匹配 Token 序列 (B, 1) -> (B, T)
        # 注意：inputs["advantages"] 通常是 (B) 或 (B, 1)
        advantages_broad = inputs["advantages"]
        if advantages_broad.dim() == 1:
            advantages_broad = advantages_broad.unsqueeze(1)
        
        # 3. 定义敏感度系数 lambda (建议 0.5 ~ 1.0，太大会导致梯度方差过大)
        conf_sensitivity = 2.0 
        
        # 4. 根据 Advantage 正负构建动态权重
        # logic: positive_adv -> weight += (1 - p)
        #        negative_adv -> weight += p
        # 使用 torch.where 实现条件选择
        
        # 判断好坏结果 (广播到序列维度)
        is_positive = (advantages_broad > 0).expand_as(token_probs)
        
        # 计算基础动态项
        dynamic_term = torch.where(
            is_positive,
            1.0 - token_probs,  # 好结果：概率越低(越不确定)，项越大
            token_probs         # 坏结果：概率越高(越自信)，项越大
        )
        
        # 生成最终的置信度权重矩阵 (Base 1.0 + 动态项)
        conf_weights = 1.0 + (conf_sensitivity * dynamic_term)
        
        # 5. 将动态权重应用到 loss_mask 上
        loss_mask = loss_mask * conf_weights
        # =======================================================================================

        # 检查是否存在 token_pos_weights (在 __init__ 中定义的)
        if hasattr(self, "token_pos_weights") and self.token_pos_weights is not None:
            # 1. 转换设备和精度以匹配 loss
            pos_weights = self.token_pos_weights.to(device=per_token_loss.device, dtype=per_token_loss.dtype)
            
            # 2. 将权重应用到 loss_mask 的末尾
            # 假设生成长度固定为 4，我们对齐序列的最后 gen_len 位
            gen_len = len(pos_weights)
            seq_len = loss_mask.shape[1]
            
            if seq_len >= gen_len:
                # 广播乘法：(Batch, gen_len) *= (gen_len,)
                loss_mask[:, -gen_len:] *= pos_weights
            else:
                # 容错处理：如果实际序列比权重短，截取权重的后半部分
                loss_mask *= pos_weights[-seq_len:]

        # ➕ [Modified] Use 'loss_mask' instead of 'mask' for loss calculation
        # 注意：分母也要变成 weighted sum，这样才是加权平均
        loss1 = ((per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)).mean()
        loss1 = loss1 / self.current_gradient_accumulation_steps
        
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            # DR_GRPO 通常分母是固定长度，这里暂时保持 mask，或者你也想加权？
            # 建议 DR_GRPO 也用加权后的 loss_mask，保持逻辑一致
            loss = (per_token_loss * loss_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * loss_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())
        return loss

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

    reward_runner = RewardRunner(
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
    trainer = GRPO_Eval_Trainer_Confidence_Aware(
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