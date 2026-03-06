#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$SCRIPT_DIR/sft_config.yaml}"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cd "$SCRIPT_DIR"
python prepare_sft_data.py --config "$CONFIG_PATH"
python train_full_sft.py --config "$CONFIG_PATH"
python evaluate_full_sft.py --config "$CONFIG_PATH"
