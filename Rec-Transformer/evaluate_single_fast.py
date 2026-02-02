import os
import json
import logging
import yaml
import argparse
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM,
    LogitsProcessorList,
    AutoTokenizer,
    LlamaForCausalLM, LlamaConfig,
    Qwen2ForCausalLM, Qwen2Config,
)

# === 导入项目依赖 (确保路径正确) ===
sys.path.append("../")
from llamarec import LlamaRecForCausalLM, LlamaRecConfig
from sasrec import SasRecForCausalLM, SasRecConfig
from utils.datacollator import EvalDataCollator, preprocess_function
from utils.utils_evaluate import (
    build_item_token_codebooks_dynamically, 
    DynamicHierarchicalLogitsProcessor,
)
from utils.eval import compute_hr_at_k, compute_ndcg_at_k
from utils.tokenizer_utils import create_pure_id_qwen_tokenizer

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class StandaloneEvaluator:
    def __init__(self, model, tokenizer, generation_config_params, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gen_len = generation_config_params['generation_length']
        self.num_beams = generation_config_params['num_beams']
        self.k_values = generation_config_params['k_values']
        self.item_token_codebooks = generation_config_params['item_token_codebooks']

        # --- 复刻 CustomTrainer 的 NumPy 向量化查找表逻辑 ---
        logger.info(">>> Building NumPy Vectorized Vocab Lookup Table for fast eval...")
        vocab = tokenizer.get_vocab()
        max_id = max(vocab.values())
        self.vocab_array = np.array(["" for _ in range(max_id + 1)], dtype=object)
        for k, v in vocab.items():
            self.vocab_array[v] = k
        logger.info("✅ Vocab Table built.")

    def evaluate(self, eval_dataloader):
        self.model.eval()
        
        total_metrics_sum = {f"HR@{k}": 0.0 for k in self.k_values}
        total_metrics_sum.update({f"NDCG@{k}": 0.0 for k in self.k_values})
        total_samples = 0

        logger.info(f"***** Running Evaluation *****")
        logger.info(f"  Num examples = {len(eval_dataloader.dataset)}")
        logger.info(f"  Batch size = {eval_dataloader.batch_size}")

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                groundtruth = batch['groundtruth']

                batch_size = input_ids.shape[0]
                prompt_length = input_ids.shape[1]

                # 1. 构建 Logits Processor
                logits_processor = LogitsProcessorList([
                    DynamicHierarchicalLogitsProcessor(
                        prompt_length=prompt_length,
                        item_token_codebooks=self.item_token_codebooks,
                        device=self.device
                    )
                ])

                # 2. 生成 (Generate)
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=prompt_length + self.gen_len,
                    num_beams=self.num_beams,
                    do_sample=False, # Eval 时通常固定为 False
                    num_return_sequences=self.num_beams,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    logits_processor=logits_processor, 
                    use_cache=True
                )

                # 3. 向量化解码 (Vectorized Decoding)
                new_tokens_cpu = generated_ids[:, -self.gen_len:].cpu().numpy()
                token_strs = self.vocab_array[new_tokens_cpu] # O(1) 查表

                # 字符串拼接
                if self.gen_len == 1:
                    predicted_token_sequences = token_strs.flatten().tolist()
                else:
                    # 使用 numpy 的字符串拼接能力
                    result_array = token_strs[:, 0]
                    for i in range(1, self.gen_len):
                        result_array = result_array + token_strs[:, i]
                    predicted_token_sequences = result_array.tolist()

                # Reshape 为 [Batch, Beam]
                reshaped_token_sequences = [
                    predicted_token_sequences[i : i + self.num_beams]
                    for i in range(0, len(predicted_token_sequences), self.num_beams)
                ]

                # 4. 计算指标 (Metrics)
                batch_hr = compute_hr_at_k(reshaped_token_sequences, groundtruth, self.k_values)
                batch_ndcg = compute_ndcg_at_k(reshaped_token_sequences, groundtruth, self.k_values)

                for k_val in self.k_values:
                    total_metrics_sum[f"HR@{k_val}"] += batch_hr[f"HR@{k_val}"] * batch_size
                    total_metrics_sum[f"NDCG@{k_val}"] += batch_ndcg[f"NDCG@{k_val}"] * batch_size
                
                total_samples += batch_size

        # 汇总
        if total_samples == 0:
            return {k: 0.0 for k in total_metrics_sum.keys()}
        
        metrics = {k: (v / total_samples) for k, v in total_metrics_sum.items()}
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Standalone Evaluation for LlamaRec/SasRec")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint folder (containing config.json and model.safetensors)")
    parser.add_argument("--dataset", type=str, default='Beauty', help="Dataset name used in config path")
    parser.add_argument("--model_name", type=str, default='llamarec', help="Model name used in config path")
    parser.add_argument("--split", type=str, default='test', choices=['train', 'test', 'valid'], help="Which split to evaluate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--output_file", type=str, default=None, help="Where to save metrics json")
    args = parser.parse_args()

    # 1. 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 2. 读取原始 Config (用于获取数据路径、Token配置等)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_script_dir, "pretrain_config", args.dataset, f"{args.model_name}.yaml")
    
    logger.info(f"Loading original training config from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    paths_config = config_data['paths']
    tokenizer_params = config_data['tokenizer_params']
    training_args = config_data['training_args']
    testing_args = config_data['testing_args']
    model_params = config_data['model_params']

    dataset_path = paths_config['dataset_path']
    tokenizer_dir = paths_config['tokenizer_dir']
    max_seq_length = model_params['max_seq_length']
    
    # 获取 generation length
    codeword_nums = tokenizer_params.get('codeword_nums', [20, 20, 20])
    generation_length = len(codeword_nums)

# 3. 加载 Tokenizer
    logger.info(f"Loading Tokenizer from checkpoint: {args.checkpoint_path}")
    try:
        # 尝试直接从 Checkpoint 文件夹加载
        # trust_remote_code=True 是为了防止如果是自定义 Tokenizer 代码
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"⚠️ Failed to load tokenizer directly from checkpoint: {e}")
        logger.warning("🔄 Falling back to rebuilding tokenizer from config...")
        
        # 如果加载失败（比如文件缺失），则回退到重建逻辑
        tokenizer = create_pure_id_qwen_tokenizer(
            output_dir=tokenizer_dir,
            codeword_nums=codeword_nums
        )

    # 再次确认关键配置（双重保险）
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None: 
        # 如果读取出来的 tokenizer 没记录 pad_token，尝试手动修复
        # 注意：这里需要小心，不要覆盖了正确的 ID，通常重建时才需要这步
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if tokenizer.bos_token_id is None: tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if tokenizer.eos_token_id is None: tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    # 4. 加载模型 Checkpoint
    logger.info(f"Loading Model from checkpoint: {args.checkpoint_path}")
    
    # 动态选择模型类
    if args.model_name == 'sasrec':
        model_class = SasRecForCausalLM
    elif args.model_name == 'llama':
        model_class = LlamaForCausalLM
    elif args.model_name.startswith('qwen'):
        model_class = Qwen2ForCausalLM
    else:
        model_class = LlamaRecForCausalLM

    try:
        model = model_class.from_pretrained(
            args.checkpoint_path, 
            torch_dtype=torch.bfloat16 if config_data['training_args'].get('bf16', False) else "auto",
            device_map=device
        )
    except Exception as e:
        logger.error(f"Failed to load model from {args.checkpoint_path}. Error: {e}")
        # 尝试自动回退
        logger.info("Trying AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, device_map=device)

    model.eval()

    # 5. 准备数据
    logger.info(f"Loading dataset from {dataset_path}, split={args.split}")
    try:
        raw_dataset = load_dataset("json", data_dir=dataset_path, split=args.split)
    except Exception:
        logger.warning(f"Split '{args.split}' not found, falling back to 'train' just to test flow.")
        raw_dataset = load_dataset("json", data_dir=dataset_path, split='train')

    # Tokenize 数据
    logger.info("Tokenizing dataset...")
    eval_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=8,
        load_from_cache_file=True, # 利用缓存
        remove_columns=['prompt'], # 保留 ground_truth
        fn_kwargs={"tokenizer": tokenizer, "max_seq_length": max_seq_length},
        desc="Tokenizing"
    )

    # Collator & DataLoader
    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=max_seq_length)
    
    batch_size = args.batch_size if args.batch_size else training_args.get('per_device_eval_batch_size', 16)
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=eval_collator,
        shuffle=False,
        drop_last=False,
        num_workers=16
    )

    # 6. 准备生成参数
    item_token_codebooks = build_item_token_codebooks_dynamically(tokenizer, generation_length)
    
    generation_config_params = {
        "generation_length": generation_length,
        "num_beams": testing_args['num_beams'],
        "k_values": testing_args['eval_k_values'],
        "item_token_codebooks": item_token_codebooks
    }

    # 7. 开始评估
    evaluator = StandaloneEvaluator(model, tokenizer, generation_config_params, device)
    metrics = evaluator.evaluate(eval_dataloader)

    # 8. 输出结果
    print("\n" + "="*30)
    print(" >>> Final Evaluation Results <<<")
    print("="*30)
    print(json.dumps(metrics, indent=4))
    print("="*30 + "\n")

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {args.output_file}")

if __name__ == "__main__":
    main()