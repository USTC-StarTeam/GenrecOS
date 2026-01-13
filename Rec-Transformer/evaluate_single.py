import os
import logging
from datasets import load_dataset
import yaml
import argparse
from typing import List
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from llamarec import LlamaRecForCausalLM, LlamaRecConfig
from sasrec import SasRecForCausalLM, SasRecConfig
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from util.datacollator import EvalDataCollator
from util.utils_evaluate import build_item_token_codebooks_dynamically, beamsearch_prefix_constraint_fn
from util.eval import compute_hr_at_k, compute_ndcg_at_k
logging.basicConfig(level=logging.INFO)



def main():
    # 获取配置文件路径
    parser = argparse.ArgumentParser(description="Train a LlamaRec model using a YAML config file.")
    parser.add_argument("--dataset", type=str, default='KuaiRand_27K_pt')
    parser.add_argument("--model_name", type=str, default='llamarec')
    parser.add_argument("--checkpoint", type=str, default='experiment/KuaiRand_27K_pt/llama-rec_20251212_013006/checkpoint-20000')
    args = parser.parse_args()

    # 读取并解析 YAML 配置文件
    logging.info(f"Loading configuration from: {args.dataset}_{args.model_name}")
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
    dataset_path = os.path.join(paths_config['dataset_path'], 'train.json')
    tokenizer_dir = paths_config['tokenizer_dir']
    max_seq_length = model_params['max_seq_length']
    generation_length = tokenizer_params['codebook_num']

    checkpoint_path = args.checkpoint
    output_dir = os.path.dirname(checkpoint_path)

    # 直接从checkpoint加载tokenizer
    logging.info(f"Loading tokenizer from checkpoint: {checkpoint_path}")
    try:
        # tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_path)
        # 使用 AutoTokenizer，并显式指定 use_fast=True（如果需要）和 trust_remote_code=True（防止自定义模型报错）
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True, trust_remote_code=True)
        logging.info("Tokenizer loaded successfully from checkpoint.")

    except Exception as e:
        logging.warning(f"Failed to load tokenizer from checkpoint: {e}")
        # 回退到从tokenizer_dir加载
        logging.info("Falling back to loading tokenizer from tokenizer_dir...")
        # tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True, trust_remote_code=True)
            logging.info("Tokenizer loaded successfully from tokenizer_dir.")
        except Exception as fallback_error:
            logging.error(f"Critical Error: Failed to load tokenizer from both locations. Error: {fallback_error}")
            raise fallback_error

    tokenizer.padding_side = 'left'

    # 补丁：确保 pad_token 存在 (Qwen 等模型有时默认没有 pad_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logging.info(f"Pad token was None, set to EOS token id: {tokenizer.pad_token_id}")

    # 数据集加载
    test_dataset = load_dataset("json", data_files=dataset_path, split='train')

    # 直接从checkpoint加载模型
    logging.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # 根据 model_name 决定使用哪个类
    if args.model_name == 'sasrec':
        model_class = SasRecForCausalLM
        config_class = SasRecConfig
        # SasRec 使用 layer_norm_eps
        norm_eps_key = 'layer_norm_eps'
        norm_eps_val = model_params.get('layer_norm_eps', 1e-12)
    else:
        model_class = LlamaRecForCausalLM
        config_class = LlamaRecConfig
        # LlamaRec 使用 rms_norm_eps
        norm_eps_key = 'rms_norm_eps'
        norm_eps_val = model_params.get('rms_norm_eps', 1e-6)

    try:
        model = model_class.from_pretrained(checkpoint_path)
        logging.info(f"Model ({args.model_name}) loaded successfully from checkpoint")
    except Exception as e:
        logging.error(f"Failed to load model from checkpoint: {e}")
        # 如果失败，回退到创建新模型（但权重不同）
        logging.warning("Creating new model architecture (weights will be random!)")
        
        config_kwargs = {
            "hidden_size": model_params['hidden_size'],
            "intermediate_size": model_params['intermediate_size'],
            "num_hidden_layers": model_params['num_hidden_layers'],
            "num_attention_heads": model_params['num_attention_heads'],
            "max_position_embeddings": max_seq_length + generation_length,
            "model_type": model_params.get('MODEL_TYPE', args.model_name),
            "vocab_size": len(tokenizer),
            "use_cache": False,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        # 注入特定的 Norm 参数
        config_kwargs[norm_eps_key] = norm_eps_val
        
        config = config_class(**config_kwargs)
        model = model_class(config)

    # try:
    #     model = LlamaRecForCausalLM.from_pretrained(checkpoint_path)
    #     logging.info(f"Model loaded successfully from checkpoint")
    # except Exception as e:
    #     logging.error(f"Failed to load model from checkpoint: {e}")
    #     # 如果失败，回退到创建新模型（但权重不同）
    #     logging.warning("Creating new model architecture (weights will be random!)")
    #     config = LlamaRecConfig(
    #         hidden_size=model_params['hidden_size'],
    #         intermediate_size=model_params['intermediate_size'],
    #         num_hidden_layers=model_params['num_hidden_layers'],
    #         num_attention_heads=model_params['num_attention_heads'],
    #         max_position_embeddings=max_seq_length + generation_length,
    #         rms_norm_eps=model_params['rms_norm_eps'],
    #         model_type=model_params['MODEL_TYPE'],
    #         vocab_size=len(tokenizer),
    #         use_cache=False,
    #         pad_token_id=tokenizer.pad_token_id,
    #         bos_token_id=tokenizer.bos_token_id,
    #         eos_token_id=tokenizer.eos_token_id,
    #     )
    #     model = LlamaRecForCausalLM(config)
    
    # 将模型移到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # DataCollator实例化
    test_collator = EvalDataCollator(tokenizer=tokenizer, max_length=max_seq_length)

    # 评估
    logging.info("Starting custom evaluation...")
    model.eval()
    
    # 构建评估 DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=training_args_dict['per_device_eval_batch_size'],
        collate_fn=test_collator,
        shuffle=False,
    )

    # 构造 Item Token 约束码本
    item_token_codebooks = build_item_token_codebooks_dynamically(tokenizer, generation_length)

    k_values=testing_args['eval_k_values']
    num_beams=testing_args['num_beams']
    total_metrics_sum = {f"HR@{k}": 0.0 for k in k_values}
    total_metrics_sum.update({f"NDCG@{k}": 0.0 for k in k_values})
    total_samples = 0
    logging.info(f"Starting manual evaluation with num_beams={num_beams}...")

    with torch.no_grad():
        logging.info("Starting manual evaluation loop...")
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # 1. 准备输入
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            groundtruth = batch['groundtruth']

            prompt_length = input_ids.shape[1] 
            # 定义beamsearch约束的闭包函数
            def batch_beamsearch_prefix_constraint_fn(batch_id: int, input_ids_tensor: torch.Tensor) -> List[int]:
                # 调用上面定义的beamsearch约束函数，并传入所有捕获的参数
                # 注意：batch_id 在这里通常被忽略，因为我们是对所有样本应用相同的约束
                return beamsearch_prefix_constraint_fn(
                    batch_id=batch_id,
                    input_ids_tensor=input_ids_tensor,
                    prompt_length=prompt_length,
                    generation_length=generation_length,
                    item_token_codebooks=item_token_codebooks # 捕获的约束列表
                )
            # 2. 预测下一个 item 的 tokens, 使用自定义的 beam search 约束来确保只生成有效的 item tokens
            generated_ids = model.generate( # (batch_size * num_beams, total_length)
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + generation_length, 
                num_beams=num_beams,
                do_sample=False, # 采样解码
                num_return_sequences=num_beams, # 返回 num_beams 个序列
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=batch_beamsearch_prefix_constraint_fn
            )
            
            # 3. 解码并提取推荐列表
            # 提取新生成的 tokens (最后 generation_length 个 tokens)
            new_tokens = generated_ids[:, -generation_length:] # (batch_size * num_beams, generation_length)
            predicted_token_sequences = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
            reshaped_token_sequences = [
                predicted_token_sequences[i : i + num_beams]
                for i in range(0, len(predicted_token_sequences), num_beams)
            ]
            # 调用自定义评估函数
            current_batch_size = len(reshaped_token_sequences)
            batch_hr = compute_hr_at_k(reshaped_token_sequences, groundtruth, k_values)
            batch_ndcg = compute_ndcg_at_k(reshaped_token_sequences, groundtruth, k_values)
            
            for k_val in k_values:
                total_metrics_sum[f"HR@{k_val}"] += batch_hr[f"HR@{k_val}"] * current_batch_size
                total_metrics_sum[f"NDCG@{k_val}"] += batch_ndcg[f"NDCG@{k_val}"] * current_batch_size
            
            total_samples += current_batch_size
            
    metrics = {name: (val / total_samples) for name, val in total_metrics_sum.items()}
    # 记录参数和指标
    log_file = os.path.join(output_dir, "evaluate_log.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"{checkpoint_path} metrics:\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value}\n")
        f.write(f"testing_args:\n")
        for key, value in tokenizer_params.items():
            f.write(f"  {key}: {value}\n")
        for key, value in testing_args.items():
            f.write(f"  {key}: {value}\n")
    logging.info(f"Evaluation results: {metrics}")
    logging.info(f"📁 Results saved to: {log_file}")

    logging.info("All operations complete!")

if __name__ == "__main__":
    main()