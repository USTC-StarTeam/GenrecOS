# train_grpo.py
import sys
import os
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
    # ================= 加载 DIN Scorer =================
    print(">>> Initializing DIN Reward Model...")
    # 你的 DIN 数据文件夹
    DIN_fuxictr_DATA_DIR = "../FuxiCTR-data/Beauty"
    DIN_fuxictr_cinfig_DIR = './CTR_models/DIN/config'
    DIN_fuxictr_model_DIR = './CTR_models/DIN/checkpoints/Beauty_onerec_think'
    DIN_RL_DATA_DIR = '../Data/Beauty/RL_data/1_rl_data_json'

    din_scorer = DINScorer(
        config_dir=DIN_fuxictr_cinfig_DIR,            # DIN 训练时的 config 目录
        model_dir=DIN_fuxictr_model_DIR,
        experiment_id='DIN_Beauty_onerec_think',      # 你的实验 ID
        data_dir=DIN_fuxictr_DATA_DIR,
        device='cuda:0'                   
    )
    print(">>> DIN Scorer Ready.")

    reward_runner = DINRewardRunner(scorer=din_scorer, weight=0.8)

    # ================= 数据集加载 =================
    # 1. 加载 Train
    train_json_path = os.path.join(DIN_RL_DATA_DIR, "train.json")
    print(f">>> Loading Train Dataset: {train_json_path}")
    train_dataset = load_dataset("json", data_files=train_json_path, split='train')

    # 2. ⚠️ 加载 Valid (用于 evaluate 和 Early Stopping)
    test_json_path = os.path.join(DIN_RL_DATA_DIR, "test.json")
    print(f">>> Loading Test Dataset: {test_json_path}")
    test_dataset = load_dataset("json", data_files=test_json_path, split='train')

    # ================= 训练配置 =================

    # 1. 加载我们在上一步生成的 RL 数据集 (JSON)
    train_json_path = os.path.join(DIN_RL_DATA_DIR, "train.json")
    print(f">>> Loading Dataset from {train_json_path}...")
    dataset = load_dataset("json", data_files=train_json_path, split="train")

    # 2. LLM 模型路径
    llm_model_path = '../Rec-Transformer/temp_experiment/Beauty/llama-rec_20260111_091208/best_model'

    EVAL_SAVE_STEPS = 500

    base_dir = "temp_try_GRPO_Rec_Output"
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_time_dir = 'checkpoints_' + date_time
    output_dir = os.path.join(base_dir, date_time_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 3. GRPO 参数
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-5,
        num_train_epochs=10,
        per_device_train_batch_size=20,
        gradient_accumulation_steps=1,
        
        # --- 训练日志与保存 ---
        logging_steps=50,
        max_completion_length=4, # RL 训练时的长度 (Training)
        num_generations=20,
        use_vllm=False,
        bf16=True,

        # --- ⚠️ 评估与早停配置 ---
        eval_strategy="steps",       # 按步数评估
        eval_steps=EVAL_SAVE_STEPS,       # 每 EVAL_STEPS 步评估一次
        per_device_eval_batch_size=20, # 评估时的 Batch Size

        save_strategy="steps",       # 按步数保存
        save_steps=EVAL_SAVE_STEPS,       # 保存频率 = 评估频率 (这样每次保存都有分数为据)
        save_total_limit=20,          # 最多保留 2 个 Checkpoint
        
        load_best_model_at_end=True, # 训练结束后加载最好的模型
        metric_for_best_model="eval_NDCG@10", # 以 NDCG@10 为标准选最好的
        greater_is_better=True,      # 指标越大越好
    )

    # 4. 加载模型与Tokenizer
    print(">>> Loading LLM...")
    model = LlamaRecForCausalLM.from_pretrained(llm_model_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)

    # Tokenizer 补丁
    if tokenizer.model_input_names is not None and "token_type_ids" in tokenizer.model_input_names:
        tokenizer.model_input_names.remove("token_type_ids")
    # 确保 padding token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ================= 准备评估用的 Codebook =================
    # 假设每个 Item ID 由 4 个 token 组成 (根据你的 max_completion_length 推测)
    # 如果你的 max_completion_length=4 是刚好生成完，那么 generation_length 应该是 4
    generation_length = 4 
    
    # 动态构建 codebook
    item_token_codebooks = build_item_token_codebooks_dynamically(tokenizer, generation_length)
    
    eval_config = {
        "generation_length": generation_length,
        "num_beams": 20,   # 评估时使用 Beam Search (通常比 greedy 好)
        "k_values": [1, 5, 10, 20],
        "item_token_codebooks": item_token_codebooks
    }

    # ================= 初始化 Trainer =================
    trainer = GRPO_Eval_Trainer(
        model=model,
        reward_funcs=[reward_runner],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,      # 传入验证集
        generation_config_params=eval_config, # 传入评估配置
        processing_class=tokenizer,
        
        # ⚠️ 添加早停回调
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=10  # 如果 HR@10 连续 3 次(3*500 step)不上升就停止
        )]
    )

    # 6. 开始训练
    print(">>> Starting GRPO Training with Live Evaluation...")
    trainer.train()

    # 7. 保存最终模型
    # 由于开启了 load_best_model_at_end，此时内存里的 model 就是最好的
    trainer.save_model("temp_try_GRPO_Rec_Output/final_best_grpo_model")
    print(">>> Training Finished & Best Model Saved.")