#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

# python expand_vocab.py \
#     --base_model_path ../models/Qwen3-1-7B \
#     --special_tokens_path ../../Data/Beauty_onerec_think/FT_data/beauty/special_tokens.json \
#     --output_path ../models/expanded_model

# nohup deepspeed ./train_align.py train_align.yaml >> train_align.log 2>&1 &

# nohup deepspeed --master_port 29600 ./train_rec.py train_rec.yaml >> train_rec.log 2>&1 &

nohup deepspeed --master_port 29700 ./train_cot.py train_cot.yaml >> train_cot.log 2>&1 &
