# 11_seq2pat_memory_prompt

This experiment mines sequential behavior patterns with `sequential.seq2pat` and injects matched patterns into SFT prompts.

## Pipeline

1. `prepare_sft_data_with_pattern_memory.py`
   - parses filtered user sequences
   - mines global sequence patterns from histories excluding each user's last test interaction
   - matches patterns to each sample history by subsequence matching only
   - builds memory-augmented prompts for train/val/test
2. `train_full_sft_pattern.py`
   - full-parameter SFT from base Qwen
3. `evaluate_full_sft_pattern.py`
   - next-item ranking evaluation via title generation + embedding retrieval
4. `summarize_results.py`
   - aggregates key metrics from outputs

## Launch

```bash
/home/kfwang/miniconda3/envs/onerec/bin/python manage_jobs.py --allowed_gpus 4,5,6,7
```

For smoke tests:

```bash
/home/kfwang/miniconda3/envs/onerec/bin/python manage_jobs.py --allowed_gpus 4 --max_users 200
```
