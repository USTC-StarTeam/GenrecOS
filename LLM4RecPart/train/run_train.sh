#!/bin/bash

export MASTER_PORT=$(shuf -n 1 -i 29500-65535)
export CUDA_VISIBLE_DEVICES=1,4

# python expand_vocab.py \
#     --base_model_path ../models/Qwen3-1-7B \
#     --special_tokens_path ../../Data/Beauty/FT_data/beauty/special_tokens.json \
#     --output_path ../models/expanded_model

# nohup deepspeed ./train_align.py train_align.yaml >> train_align.log 2>&1 &

nohup deepspeed --num_gpus 2 --master_port ${MASTER_PORT} ./train_align_parallel.py train_align_parallel.yaml >> train_align_parallel.log 2>&1 &

# nohup deepspeed --master_port 29600 ./train_rec.py train_rec.yaml >> train_rec.log 2>&1 &

nohup deepspeed --master_port 29700 ./train_cot.py train_cot.yaml >> train_cot.log 2>&1 &
