# SASRec Training for Beauty Dataset

This directory contains scripts to train SASRec (Self-Attentive Sequential Recommendation) on the Amazon Beauty dataset using vanilla item IDs.

## Directory Structure

```
vanilla_sasrec/
├── 0000_detect_sequence_length.py    # Analyze sequence length statistics
├── 0001_preprocess_beauty_data.py    # Convert jsonl to train/val/test format
├── 0002_inspect_processed_data.py    # Verify processed data
├── config.yaml                        # Training configuration
├── train_sasrec.py                    # Main training script
├── evaluate_sasrec.py                 # Evaluation script
├── run.sh                             # Run all steps
├── processed_data/                    # Generated after preprocessing
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   ├── item_mapping.json
│   └── splits/
└── checkpoints/                       # Model checkpoints (generated during training)
```

## Dataset Statistics

After preprocessing (min_item_freq=5, min_seq_length=3):
- **Users**: 5,312 users with >=3 interactions
- **Items**: 28,344 items with >=5 interactions
- **Train samples**: 12,744 (sliding window sequences)
- **Val samples**: 5,312 (second-to-last item prediction)
- **Test samples**: 5,312 (last item prediction)

Sequence length statistics:
- Mean: 6.11 items
- Max: 115 items
- Median: ~3 items

## Quick Start

### 1. Run all steps at once

```bash
bash run.sh
```

### 2. Or run steps individually

```bash
# Step 1: Analyze data
python 0000_detect_sequence_length.py

# Step 2: Preprocess data
python 0001_preprocess_beauty_data.py

# Step 3: Inspect processed data (optional)
python 0002_inspect_processed_data.py

# Step 4: Train model
python train_sasrec.py --config config.yaml
```

### 3. Evaluate a trained model

```bash
python evaluate_sasrec.py \
    --model_path ./checkpoints/sasrec_beauty_XXX/best_model \
    --data_path ./processed_data/splits \
    --num_beams 10 \
    --k_values 1 5 10 20
```

## Configuration

Edit `config.yaml` to adjust training parameters:

### Model Architecture
- `max_seq_length`: Maximum input sequence length (default: 100)
- `hidden_size`: Transformer hidden size (default: 128)
- `num_hidden_layers`: Number of transformer layers (default: 2)
- `num_attention_heads`: Number of attention heads (default: 4)

### Training
- `learning_rate`: Learning rate (default: 1e-3)
- `num_train_epochs`: Number of training epochs (default: 50)
- `per_device_train_batch_size`: Batch size (default: 64)

### Evaluation
- `num_beams`: Beam size for generation (default: 10)
- `eval_k_values`: K values for HR@K and NDCG@K (default: [1, 5, 10, 20])

## Data Format

Each sample in the processed data has the format:
```json
{
    "prompt": "7858 4345 2695 ...",
    "ground_truth": "12345",
    "user_id": "A2XFO5VBQTWDRN"
}
```

- `prompt`: Space-separated item IDs representing user history
- `ground_truth`: Target item ID to predict
- `user_id`: Original Amazon user ID (for reference)

## Notes

1. **Sparse Data**: This dataset is very sparse with most users having short sequences (mean ~6 items)

2. **Item Filtering**: Items with frequency < 5 are filtered out, resulting in 28K items from original 115K

3. **Chronological Split**: Data is split chronologically with:
   - Last item → test
   - Second-to-last → validation
   - Remaining → training (with sliding window augmentation)

4. **Tokenizer**: Simple item ID tokenizer where each item is a single token (e.g., "12345")

5. **Evaluation**: Uses beam search with constrained generation to predict next item
