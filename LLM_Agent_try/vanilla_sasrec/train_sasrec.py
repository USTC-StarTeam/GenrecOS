#!/usr/bin/env python3
"""
SASRec Training Script for Beauty Dataset
Adapted from Rec-Transformer/train_single.py for vanilla item IDs

Usage:
    python train_sasrec.py --config config.yaml
"""

import os
import json
import logging
import yaml
import argparse
import sys
import warnings
import random
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    LogitsProcessorList,
    PreTrainedTokenizerFast,
)
import transformers.utils.logging
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Add parent directories to path for imports
sys.path.append("../../Rec-Transformer")
sys.path.append("../..")

# Import SASRec model
from sasrec import SasRecForCausalLM, SasRecConfig

# Import utils
from utils.datacollator import TrainDataCollator, EvalDataCollator, preprocess_function
from utils.utils_evaluate import (
    build_item_token_codebooks_dynamically,
    beamsearch_prefix_constraint_fn,
    DynamicHierarchicalLogitsProcessor,
)
from utils.eval import compute_hr_at_k, compute_ndcg_at_k
from utils.utils import *

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.trainer")


# =============================================================================
# Custom Trainer with Generation-based Evaluation
# =============================================================================
class SASRecTrainer(Trainer):
    def __init__(self, eval_collator, generation_config_params, **kwargs):
        super().__init__(**kwargs)
        self.eval_collator = eval_collator
        self.gen_len = generation_config_params['generation_length']
        self.num_beams = generation_config_params['num_beams']
        self.k_values = generation_config_params['k_values']
        self.vocab_id_to_item = generation_config_params.get('vocab_id_to_item', {})

        # Build NumPy vectorized vocab lookup table
        vocab = kwargs['processing_class'].get_vocab()
        max_id = max(vocab.values())
        self.vocab_array = np.array(["" for _ in range(max_id + 1)], dtype=object)
        for k, v in vocab.items():
            self.vocab_array[v] = k

        logging.info("✅ NumPy Vectorized Vocab Lookup Table built.")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        target_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        # Sampling for validation speed
        eval_sample_num = 2000
        if metric_key_prefix == "eval" and target_dataset is not None:
            total_size = len(target_dataset)
            if total_size > eval_sample_num:
                logging.info(f"⚡ Sampling {eval_sample_num} from {total_size} for validation.")
                random_indices = random.sample(range(total_size), eval_sample_num)
                target_dataset = target_dataset.select(random_indices)
            else:
                logging.info(f"Dataset size ({total_size}) <= {eval_sample_num}, running full evaluation.")

        eval_dataloader = DataLoader(
            target_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.eval_collator,
            shuffle=False,
            drop_last=False
        )

        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()

        logging.info(f"***** Running Custom Evaluation ({metric_key_prefix}) *****")
        logging.info(f"  Num examples = {len(target_dataset)}")
        logging.info(f"  Batch size = {self.args.eval_batch_size}")

        total_metrics_sum = {f"HR@{k}": 0.0 for k in self.k_values}
        total_metrics_sum.update({f"NDCG@{k}": 0.0 for k in self.k_values})
        total_samples = 0

        max_k = max(self.k_values)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc=f"Evaluating ({metric_key_prefix})")):
                input_ids = batch['input_ids'].to(self.args.device)
                attention_mask = batch['attention_mask'].to(self.args.device)
                groundtruth = batch['groundtruth']

                batch_size = input_ids.shape[0]

                # Forward pass to get logits
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]

                # Get logits for the last position (next item prediction)
                last_logits = logits[:, -1, :]  # [batch_size, vocab_size]

                # Get top-K predictions (excluding special tokens: PAD=0, UNK=1, BOS=2, EOS=3)
                # Item token IDs start from 4, so we slice [4:] and then add 4 back
                # But we need to convert token IDs to item IDs for comparison
                # Token ID = Item ID + 4 (since special tokens are 0-3)
                # So Item ID = Token ID - 4
                top_k_scores, top_k_indices = torch.topk(last_logits[:, 4:], k=max_k, dim=-1)
                # top_k_indices are indices into the sliced tensor (0 = item 0)
                # Convert to item IDs (same as indices since items are 0-indexed)
                # predictions are item IDs (0-indexed), groundtruth is also item ID string
                predictions = top_k_indices.cpu().numpy().tolist()  # [batch_size, max_k]

                # Compute metrics for each sample
                for i in range(batch_size):
                    pred_list = [str(p) for p in predictions[i]]  # Item IDs as strings
                    gt = groundtruth[i]  # Already an item ID string

                    for k_val in self.k_values:
                        # HR@K
                        if gt in pred_list[:k_val]:
                            total_metrics_sum[f"HR@{k_val}"] += 1.0
                            # NDCG@K
                            rank = pred_list[:k_val].index(gt) + 1
                            total_metrics_sum[f"NDCG@{k_val}"] += 1.0 / np.log2(rank + 1)

                total_samples += batch_size

        if total_samples == 0:
            metrics = {f"{metric_key_prefix}_{k}": 0.0 for k in total_metrics_sum.keys()}
        else:
            metrics = {f"{metric_key_prefix}_{k}": (v / total_samples) for k, v in total_metrics_sum.items()}

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        logging.info(f"Evaluation metrics: {metrics}")
        return metrics


# =============================================================================
# Simple Item ID Tokenizer
# =============================================================================
def create_item_id_tokenizer(num_items, special_tokens_map=None, output_dir="./tokenizer"):
    """
    Create a simple tokenizer for item IDs (0, 1, 2, ..., num_items-1)
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from tokenizers.processors import TemplateProcessing

    if special_tokens_map is None:
        special_tokens_map = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
        }

    # Build vocabulary
    special_tokens = [
        special_tokens_map["pad_token"],
        special_tokens_map["unk_token"],
        special_tokens_map["bos_token"],
        special_tokens_map["eos_token"]
    ]

    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    next_id = len(special_tokens)

    # Add item IDs as tokens
    for item_id in range(num_items):
        token = str(item_id)
        vocab[token] = next_id
        next_id += 1

    logging.info(f"Built vocabulary with {len(vocab)} tokens")
    logging.info(f"Item ID range: 0 to {num_items - 1}")

    # Initialize tokenizer
    custom_tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=special_tokens_map["unk_token"]))
    custom_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # Post-processor
    custom_tokenizer.post_processor = TemplateProcessing(
        single="$A",
        special_tokens=[
            (special_tokens_map["bos_token"], vocab[special_tokens_map["bos_token"]]),
            (special_tokens_map["eos_token"], vocab[special_tokens_map["eos_token"]]),
        ],
    )

    # Create PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=custom_tokenizer,
        pad_token=special_tokens_map["pad_token"],
        unk_token=special_tokens_map["unk_token"],
        bos_token=special_tokens_map["bos_token"],
        eos_token=special_tokens_map["eos_token"],
    )

    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Tokenizer saved to {output_dir}")

    return tokenizer


# =============================================================================
# Main Training Function
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train SASRec on Beauty dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    logging.info(f"Loading configuration from: {config_path}")

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    paths_config = config_data['paths']
    model_params = config_data['model_params']
    training_args_dict = config_data['training_args']
    tokenizer_params = config_data['tokenizer_params']
    testing_args = config_data['testing_args']

    # Setup paths
    dataset_path = paths_config['dataset_path']
    # Make dataset_path relative to this script
    if not os.path.isabs(dataset_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, dataset_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(paths_config['output_dir'], f"sasrec_beauty_{timestamp}")
    tokenizer_dir = paths_config['tokenizer_dir']
    if not os.path.isabs(tokenizer_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_dir = os.path.join(script_dir, tokenizer_dir)

    max_seq_length = model_params['max_seq_length']
    generation_length = testing_args['generation_length']

    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file_path = os.path.join(output_dir, "training_process.log")

    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logging.info(f"✅ Logging started. Output file: {log_file_path}")

    # Load item mapping to get vocab size
    processed_data_dir = os.path.dirname(dataset_path)
    item_mapping_path = os.path.join(processed_data_dir, "item_mapping.json")

    if os.path.exists(item_mapping_path):
        with open(item_mapping_path, 'r') as f:
            item_mapping = json.load(f)
            num_items = item_mapping['num_items']
        logging.info(f"Loaded item mapping: {num_items} items")
    else:
        # Fallback: detect from config
        num_items = tokenizer_params.get('vocab_size', 10000)
        logging.warning(f"Item mapping not found, using vocab_size: {num_items}")

    # Create tokenizer
    logging.info("Creating item ID tokenizer...")
    tokenizer = create_item_id_tokenizer(
        num_items=num_items,
        output_dir=tokenizer_dir
    )
    # Set padding side to left for decoder-only generation
    tokenizer.padding_side = 'left'

    # Load datasets
    logging.info(f"Loading datasets from {dataset_path}...")
    train_dataset = load_dataset("json", data_dir=dataset_path, split='train')
    valid_dataset = load_dataset("json", data_dir=dataset_path, split='validation')
    test_dataset = load_dataset("json", data_dir=dataset_path, split='test')

    logging.info(f"  Train: {len(train_dataset)} samples")
    logging.info(f"  Valid: {len(valid_dataset)} samples")
    logging.info(f"  Test: {len(test_dataset)} samples")

    # Create model
    logging.info("Creating SASRec model...")
    config_kwargs = {
        "vocab_size": len(tokenizer),
        "max_position_embeddings": max_seq_length + generation_length,
        "model_type": model_params.get('MODEL_TYPE', 'sasrec'),
        "use_cache": False,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Add model params
    config_kwargs.update(model_params)
    config_kwargs.pop('MODEL_TYPE', None)

    config = SasRecConfig(**config_kwargs)
    model = SasRecForCausalLM(config)

    logging.info(f"Model created with {model.num_parameters() / 1e6:.2f} M parameters.")

    # Setup training arguments
    training_args_dict['output_dir'] = output_dir
    training_args_dict['logging_dir'] = os.path.join(output_dir, 'logs')
    training_args = TrainingArguments(**training_args_dict)

    # Preprocess datasets
    logging.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=training_args_dict.get('dataloader_num_workers', 4),
        load_from_cache_file=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length
        },
        remove_columns=["prompt", 'ground_truth', 'user_id'],
        desc="Tokenizing train dataset",
    )

    valid_dataset = valid_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=training_args_dict.get('dataloader_num_workers', 4),
        load_from_cache_file=True,
        remove_columns=['prompt', 'user_id'],
        fn_kwargs={"tokenizer": tokenizer, "max_seq_length": max_seq_length},
        desc="Tokenizing valid dataset"
    )

    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=training_args_dict.get('dataloader_num_workers', 4),
        load_from_cache_file=True,
        remove_columns=['prompt', 'user_id'],
        fn_kwargs={"tokenizer": tokenizer, "max_seq_length": max_seq_length},
        desc="Tokenizing test dataset"
    )

    # Data collators
    train_collator = TrainDataCollator(tokenizer=tokenizer, max_length=max_seq_length)
    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=max_seq_length)

    # Generation config (not used for vanilla item IDs, but kept for compatibility)
    generation_config_params = {
        "generation_length": generation_length,
        "num_beams": testing_args['num_beams'],
        "k_values": testing_args['eval_k_values'],
        "vocab_id_to_item": {}
    }

    # Create trainer
    trainer = SASRecTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=train_collator,
        eval_collator=eval_collator,
        generation_config_params=generation_config_params,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=testing_args['early_stopping_patience'])]
    )

    # Train
    logging.info("Starting training...")
    trainer.train()

    # Print best results
    if trainer.state.best_model_checkpoint:
        best_metric = training_args.metric_for_best_model
        logging.info("=" * 40)
        logging.info(f"🏆 Best Model Checkpoint: {trainer.state.best_model_checkpoint}")
        logging.info(f"Best Metric ({best_metric}): {trainer.state.best_metric}")
        logging.info("=" * 40)

    # Final test evaluation
    logging.info("Starting Final Evaluation on Test Set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    test_results_path = os.path.join(output_dir, "test_results.json")
    with open(test_results_path, "w") as f:
        json.dump(test_metrics, f, indent=4)

    logging.info(f"Test results saved to {test_results_path}")

    # Save final model
    final_model_path = os.path.join(output_dir, "best_model")
    logging.info(f"Saving best model to {final_model_path}")

    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    logging.info("All operations complete!")


if __name__ == "__main__":
    main()
