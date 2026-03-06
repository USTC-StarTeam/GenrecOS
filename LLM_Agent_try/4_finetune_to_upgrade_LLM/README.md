# Qwen Full SFT on Condensed-Title Sequence Prediction

## Goal

This directory turns the earlier zero-shot LLM recommendation attempt into a supervised fine-tuning pipeline. The task is:

- input: a user’s interaction history represented by condensed item titles, ratings, and truncated reviews
- output: the next condensed item title

Unlike the earlier `LLM_baseline_Prediction/` scripts, this branch performs **full-parameter SFT** of the local `Qwen3-1-7B` checkpoint on GPU `0`.

## Data Design

The data is rebuilt from `../../Data/Amazons/data/All_Beauty.jsonl` plus condensed titles from `../use_Qwen3-1-7B_to_generate_title/item_titles_unique.json`.
It now follows the same construction logic as `../LLM_Rec_Data_Preparation/prepare_llm_rec_data.py`, but keeps the final target as the refined item title.

Rules:

- minimum item frequency: `5`
- minimum sequence length: `3`
- maximum history length: `20`
- split strategy:
  - last item -> `test`
  - second-to-last item -> `val`
  - earlier next-item targets -> `train`

Generated dataset sizes:

- train: `6,682`
- val: `4,548`
- test: `4,548`

Each sample stores both raw IDs and the final prompt text. Training uses a completion-only loss: system + user prompt are masked, and only the assistant title target contributes to loss.

## Files

- `prepare_sft_data.py`: rebuilds train/val/test JSONL files
- `train_full_sft.py`: full-parameter SFT with Hugging Face `Trainer`
- `evaluate_full_sft.py`: test-time generation plus embedding retrieval metrics
- `pipeline_utils.py`: shared prompt formatting and dataset helpers
- `sft_config.yaml`: all paths and hyperparameters
- `run_pipeline.sh`: one-shot data -> train -> eval launcher

## Training Setup

The finished run used:

- device: `CUDA_VISIBLE_DEVICES=0`
- optimizer: `adamw_torch_fused`
- dtype: `bf16`
- attention: `sdpa`
- gradient checkpointing: enabled
- train batch size: `4`
- gradient accumulation: `4`
- eval batch size: `4`
- max length: `1232`
- max history length: `20`
- max review length: `150`
- epochs: up to `5`, with early stopping

Observed training summary:

- train runtime: `3282.46s`
- train samples/s: `10.18`
- training stopped at epoch: `4.7852`
- best checkpoint: `checkpoint-1600`
- best eval loss: `3.2269`

Metrics are saved in `outputs/qwen3_title_sft/train_metrics.json`.

## Test Results

Best checkpoint: `outputs/qwen3_title_sft/best_model`

Final test metrics from `results/evaluation_metrics.json`:

- `HR@1`: `0.1282`
- `HR@5`: `0.1609`
- `HR@10`: `0.1642`
- `HR@20`: `0.1706`
- `NDCG@10`: `0.1477`
- exact title match: `0.1574`

## Interpretation

This corrected data pipeline is a clear improvement over the earlier broken title-only SFT setup in this folder.

- old broken-data SFT: `HR@1 = 0.0830`, `HR@10 = 0.1086`
- corrected-data full SFT: `HR@1 = 0.1282`, `HR@10 = 0.1642`

So the main gain came from fixing data construction to match the established recommendation split/prompt design, not from changing the base model.

## How to Re-run

```bash
cd 4_finetune_to_upgrade_LLM
bash run_pipeline.sh
```

Or step by step:

```bash
python prepare_sft_data.py --config sft_config.yaml
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True python train_full_sft.py --config sft_config.yaml
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True python evaluate_full_sft.py --config sft_config.yaml
```

## Useful Outputs

- `data/dataset_summary.json`
- `outputs/qwen3_title_sft/best_model/`
- `outputs/qwen3_title_sft/train_metrics.json`
- `results/evaluation_metrics.json`
- `results/sample_predictions.json`

## Next Likely Improvements

- improve the target format beyond plain title generation
- add stronger ranking-aware supervision instead of pure next-title completion
- inject more structured history information than a bare title list
- compare full SFT against lightweight adapters on the same split
