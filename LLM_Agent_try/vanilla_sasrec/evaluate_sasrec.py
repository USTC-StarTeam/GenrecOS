#!/usr/bin/env python3
"""
Evaluation script for SASRec model
Evaluates a trained checkpoint on test set
"""

import os
import json
import logging
import argparse
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Add parent directories to path for imports
sys.path.append("../../Rec-Transformer")
sys.path.append("../..")

# Import SASRec model
from sasrec import SasRecForCausalLM, SasRecConfig

# Import utils
from utils.datacollator import EvalDataCollator
from utils.eval import compute_hr_at_k, compute_ndcg_at_k

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_model(model_path, data_path, output_dir=None, batch_size=32, k_values=[1, 5, 10, 20], num_beams=10):
    """
    Evaluate SASRec model on test set

    Args:
        model_path: Path to trained model checkpoint
        data_path: Path to processed data directory (splits/)
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        k_values: K values for HR@K and NDCG@K
        num_beams: Number of beams for generation
    """
    logging.info("=" * 60)
    logging.info("SASRec Model Evaluation")
    logging.info("=" * 60)

    # Load model
    logging.info(f"Loading model from {model_path}...")
    model = SasRecForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load tokenizer
    logging.info(f"Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

    # Get vocab for decoding
    vocab = tokenizer.get_vocab()
    max_id = max(vocab.values())
    vocab_array = np.array(["" for _ in range(max_id + 1)], dtype=object)
    for k, v in vocab.items():
        vocab_array[v] = k

    logging.info(f"Model loaded on {device}")
    logging.info(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")

    # Load test dataset
    logging.info(f"Loading test dataset from {data_path}...")
    test_dataset = load_dataset("json", data_dir=data_path, split='test')
    logging.info(f"Test samples: {len(test_dataset)}")

    # Data collator
    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=100)

    # DataLoader
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=eval_collator,
        shuffle=False,
        drop_last=False
    )

    # Evaluation
    logging.info("Starting evaluation...")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Num beams: {num_beams}")
    logging.info(f"  K values: {k_values}")

    total_metrics_sum = {f"HR@{k}": 0.0 for k in k_values}
    total_metrics_sum.update({f"NDCG@{k}": 0.0 for k in k_values})
    total_samples = 0

    gen_len = 1  # Generate single item

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            groundtruth = batch['groundtruth']

            batch_size = input_ids.shape[0]
            prompt_length = input_ids.shape[1]

            # Generate predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=prompt_length + gen_len,
                num_beams=num_beams,
                do_sample=False,
                num_return_sequences=num_beams,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

            # Decode predictions
            new_tokens_cpu = generated_ids[:, -gen_len:].cpu().numpy()
            token_strs = vocab_array[new_tokens_cpu]
            predicted_token_sequences = token_strs.flatten().tolist()

            # Reshape for beam search
            reshaped_token_sequences = [
                predicted_token_sequences[i: i + num_beams]
                for i in range(0, len(predicted_token_sequences), num_beams)
            ]

            # Compute metrics
            batch_hr = compute_hr_at_k(reshaped_token_sequences, groundtruth, k_values)
            batch_ndcg = compute_ndcg_at_k(reshaped_token_sequences, groundtruth, k_values)

            for k_val in k_values:
                total_metrics_sum[f"HR@{k_val}"] += batch_hr[f"HR@{k_val}"] * batch_size
                total_metrics_sum[f"NDCG@{k_val}"] += batch_ndcg[f"NDCG@{k_val}"] * batch_size

            total_samples += batch_size

    # Compute final metrics
    metrics = {k: (v / total_samples) for k, v in total_metrics_sum.items()}

    logging.info("\n" + "=" * 60)
    logging.info("Evaluation Results")
    logging.info("=" * 60)
    for k in k_values:
        logging.info(f"  HR@{k:2d}:  {metrics[f'HR@{k}']:.4f}")
        logging.info(f"  NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")

    # Save results
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"\nResults saved to {results_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SASRec model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="./processed_data/splits", help="Path to test data")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_beams", type=int, default=10, help="Number of beams")
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10, 20], help="K values for metrics")

    args = parser.parse_args()

    # Make data_path relative if needed
    if not os.path.isabs(args.data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.data_path = os.path.join(script_dir, args.data_path)

    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        k_values=args.k_values,
        num_beams=args.num_beams
    )


if __name__ == "__main__":
    main()
