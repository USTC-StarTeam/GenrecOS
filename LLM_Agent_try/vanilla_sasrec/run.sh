#!/bin/bash
# Run script for SASRec training on Beauty dataset

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Set PYTHONPATH for SASRec imports
export PYTHONPATH=$PYTHONPATH:$(pwd)/../../Rec-Transformer

echo "========================================"
echo "SASRec Training for Beauty Dataset"
echo "========================================"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Step 1: Analyze data
echo "[Step 1] Analyzing data..."
python 0000_detect_sequence_length.py

echo ""
echo "[Step 2] Preprocessing data..."
python 0001_preprocess_beauty_data.py

echo ""
echo "[Step 3] Starting training..."
python train_sasrec.py --config config.yaml

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
