# train_grpo.py
import sys
import os
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ================= 配置路径 =================
# 1. 确保能导入相关模块
sys.path.append("Rec-Transformer") # LlamaRec 所在
# sys.path.append("RL-TRL/DIN") # din_evaluator 所在
# 还需要确保 FuxiCTR 在 path 里，如果在 site-packages 里则不需要加

# 导入自定义模块
from DIN.DIN_evaluator import DINScorer
from llamarec import LlamaRecConfig, LlamaRecForCausalLM 

# ================= 注册模型 =================
AutoConfig.register("llama-rec", LlamaRecConfig)
AutoModelForCausalLM.register(LlamaRecConfig, LlamaRecForCausalLM)

# ================= 加载 DIN Scorer =================
print(">>> Initializing DIN Reward Model...")
# 你的 DIN 数据文件夹
DIN_fuxictr_DATA_DIR = "FuxiCTR-data/Beauty_onerec_think"
DIN_fuxictr_cinfig_DIR = 'RL-TRL/DIN/config'
DIN_RL_DATA_DIR = 'Data/Beauty_onerec_think/RL_data/rl_data_json'

din_scorer = DINScorer(
    config_dir=DIN_fuxictr_cinfig_DIR,            # DIN 训练时的 config 目录
    experiment_id='DeepFM_test',      # 你的实验 ID
    data_dir=DIN_fuxictr_DATA_DIR,
    device='cuda:0'                   # 放在卡0，LLM 如果很大可能需要放在卡1
)
print(">>> DIN Scorer Ready.")

# ================= 定义 Reward Function =================
def reward_din_score(prompts, completions, ground_truth, user_id, **kwargs):
    """
    TRL 能够自动从 dataset 中提取对应的列作为参数传递进来。
    dataset 列名: ['prompt', 'ground_truth', 'user_id']
    所以我们在参数里加上 user_id 即可接收。
    """
    
    rewards = []
    
    # 我们先收集需要 DIN 打分的索引，避免格式错误的也去跑模型浪费时间
    to_score_indices = []
    to_score_prompts = []
    to_score_completions = []
    to_score_uids = []
    
    # 临时结果存储
    temp_scores = [0.0] * len(prompts)
    
    for i, (c, gt) in enumerate(zip(completions, ground_truth)):
        # 1. Format Reward (格式检查)
        c = c.strip()
        if not c.startswith("<") or not c.endswith(">"):
            temp_scores[i] = -1.0 # 格式错误重罚
            continue
            
        # 2. Ground Truth Reward (硬匹配)
        # 如果直接猜中，给满分 1.0
        # 注意：dataset 里 ground_truth 是 string，completions 也是 string
        if c == gt:
            temp_scores[i] = 1.0 
            continue
            
        # 3. 如果格式对但没猜中，加入 DIN 打分队列
        # 这是一个 Soft Reward
        to_score_indices.append(i)
        to_score_prompts.append(prompts[i])
        to_score_completions.append(c)
        to_score_uids.append(user_id[i]) # 注意 user_id 是一个 batch 的 list

    # 4. 批量调用 DIN
    if to_score_indices:
        try:
            model_scores = din_scorer.predict_batch(
                user_ids=to_score_uids,
                prompts=to_score_prompts,
                completions=to_score_completions
            )
            
            # 回填分数
            # 策略：DIN 输出是 Probability (0~1)
            # 我们可以给一个系数，比如 0.5，意味着即使没猜中 GT，
            # 如果 DIN 认为很匹配，最高也能拿 0.5 分
            for idx, score in zip(to_score_indices, model_scores):
                # 保护一下，防止 score 是负数(我们在 evaluator 里对未知 item 给了 -0.1)
                final_score = max(-0.1, score) 
                temp_scores[idx] = final_score * 0.8 # 系数可调
                
        except Exception as e:
            print(f"Error in DIN Scorer: {e}")
            # 出错保持 0.0
            
    return temp_scores

# ================= 训练配置 =================

# 1. 加载我们在上一步生成的 RL 数据集 (JSON)
train_json_path = os.path.join(DIN_RL_DATA_DIR, "train.json")
print(f">>> Loading Dataset from {train_json_path}...")
dataset = load_dataset("json", data_files=train_json_path, split="train")

# 2. LLM 模型路径
llm_model_path = r'/zhdd/home/kfwang/20250813Reproduct_Onerec/Fuxi-OneRec-new/Rec-Transformer/experiment/KuaiRand/checkpoint-206'

# 3. GRPO 参数
training_args = GRPOConfig(
    output_dir="try_GRPO_Rec_Output_Final",
    learning_rate=5e-7,              # 学习率通常要小
    num_train_epochs=1,
    per_device_train_batch_size=2,   # 根据显存调整，因为同时加载了 DIN 和 LLM
    gradient_accumulation_steps=4,
    logging_steps=5,
    max_completion_length=24,        # 稍微大一点，防止截断 <s_a_..><s_b_..>
    num_generations=4,               # 每次采样 4 个
    use_vllm=False,                  # 暂时关闭 vLLM 简化环境
    bf16=True,                       # 如果显卡支持 BF16 (3090/A100) 建议开启
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

# 5. 初始化 Trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_din_score],
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# 6. 开始训练
print(">>> Starting GRPO Training...")
trainer.train()

# 7. 保存最终模型
trainer.save_model("final_grpo_model")
print(">>> Training Finished & Model Saved.")