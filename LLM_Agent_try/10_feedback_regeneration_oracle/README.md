# 10_feedback_regeneration_oracle

Goal: evaluate whether an SFT model can improve recommendation by regeneration after oracle feedback.

## Method

1. Generate one next-title answer from the SFT model.
2. Retrieve top-k condensed titles by embedding similarity (`k=5` for feedback trigger).
3. If target item is not in top-5, send feedback to the model:
   - "previous answer is incorrect (target not in top-5), regenerate".
4. Repeat feedback-regeneration for at most 3 rounds.

Notes:
- This is an oracle experiment. Triggering feedback depends on target-label hit/miss and is intentionally data-leaky for validation.
- Model checkpoint defaults to `../9_sft_with_interest_summary/outputs/qwen3_interest_sft/best_model`.

## Run

```bash
python manage_jobs.py --config feedback_regen_config.yaml --allowed_gpus 0,1,2,3 --poll_seconds 20
```

## Outputs

- Prepared datasets: `data/`
- Job logs: `logs/`
- Job status: `jobs/manager_status.json`
- Per-dataset metrics: `results/*_metrics.json`
- Sample traces: `results/*_sample_traces.json`
- Summary: `results/summary.json`
