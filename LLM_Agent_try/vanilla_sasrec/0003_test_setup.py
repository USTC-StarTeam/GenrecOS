#!/usr/bin/env python3
"""
Quick test to verify the setup works
"""

import sys
import os

# Add paths
sys.path.append("../../Rec-Transformer")
sys.path.append("../..")

print("=" * 60)
print("Testing SASRec Training Setup")
print("=" * 60)

# Test 1: Import SASRec
print("\n[Test 1] Importing SASRec modules...")
try:
    from sasrec import SasRecForCausalLM, SasRecConfig
    print("  ✅ SASRec imports successful")
except Exception as e:
    print(f"  ❌ SASRec import failed: {e}")
    exit(1)

# Test 2: Import utils
print("\n[Test 2] Importing utils modules...")
try:
    from utils.datacollator import TrainDataCollator, EvalDataCollator
    from utils.eval import compute_hr_at_k, compute_ndcg_at_k
    print("  ✅ Utils imports successful")
except Exception as e:
    print(f"  ❌ Utils import failed: {e}")
    exit(1)

# Test 3: Load processed data
print("\n[Test 3] Loading processed data...")
try:
    from datasets import load_dataset
    data_path = "./processed_data/splits"
    train_dataset = load_dataset("json", data_dir=data_path, split='train')
    valid_dataset = load_dataset("json", data_dir=data_path, split='validation')
    test_dataset = load_dataset("json", data_dir=data_path, split='test')
    print(f"  ✅ Train: {len(train_dataset)} samples")
    print(f"  ✅ Valid: {len(valid_dataset)} samples")
    print(f"  ✅ Test: {len(test_dataset)} samples")
except Exception as e:
    print(f"  ❌ Data loading failed: {e}")
    exit(1)

# Test 4: Create tokenizer
print("\n[Test 4] Creating tokenizer...")
try:
    import json
    with open("./processed_data/item_mapping.json", 'r') as f:
        item_mapping = json.load(f)
    num_items = item_mapping['num_items']

    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers

    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    next_id = len(special_tokens)

    for item_id in range(num_items):
        vocab[str(item_id)] = next_id
        next_id += 1

    custom_tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    custom_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=custom_tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )

    print(f"  ✅ Tokenizer created with {len(tokenizer)} tokens")
except Exception as e:
    print(f"  ❌ Tokenizer creation failed: {e}")
    exit(1)

# Test 5: Create model
print("\n[Test 5] Creating SASRec model...")
try:
    import torch

    config = SasRecConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=100,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = SasRecForCausalLM(config)

    print(f"  ✅ Model created with {model.num_parameters() / 1e6:.2f}M parameters")

    # Test forward pass
    input_ids = torch.randint(0, len(tokenizer), (2, 10))
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss

    print(f"  ✅ Forward pass successful, loss: {loss.item():.4f}")
except Exception as e:
    print(f"  ❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Test data collator
print("\n[Test 6] Testing data collator...")
try:
    from utils.datacollator import TrainDataCollator, EvalDataCollator

    train_collator = TrainDataCollator(tokenizer=tokenizer, max_length=50)
    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=50)

    # Test train collator
    sample_batch = [
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
        {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1]},
    ]
    batch = train_collator(sample_batch)
    print(f"  ✅ Train collator: input_ids shape {batch['input_ids'].shape}")

    # Test eval collator
    eval_sample_batch = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "ground_truth": "12345"},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "ground_truth": "67890"},
    ]
    batch = eval_collator(eval_sample_batch)
    print(f"  ✅ Eval collator: input_ids shape {batch['input_ids'].shape}, groundtruth: {batch['groundtruth']}")

except Exception as e:
    print(f"  ❌ Data collator test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 7: Test generation
print("\n[Test 7] Testing model generation...")
try:
    model.eval()
    input_ids = torch.randint(0, len(tokenizer), (1, 10))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=15,
            num_beams=3,
            do_sample=False,
            num_return_sequences=3,
            pad_token_id=tokenizer.pad_token_id,
        )

    print(f"  ✅ Generation successful: shape {generated.shape}")
    print(f"     Input length: {input_ids.shape[1]}, Generated tokens: {generated.shape[1] - input_ids.shape[1]}")

except Exception as e:
    print(f"  ❌ Generation test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✅")
print("=" * 60)
print("\nYou can now run training with:")
print("  python train_sasrec.py --config config.yaml")
