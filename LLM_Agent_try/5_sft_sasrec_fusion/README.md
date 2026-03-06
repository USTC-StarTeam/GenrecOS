# 5_sft_sasrec_fusion

这个目录只做一件事：把 `4_finetune_to_upgrade_LLM/` 训练出来的 SFT 模型，拿来和 `vanilla_sasrec/` 再做一次融合。

当前目录内包含两条主线：

- `eval_fixed_fusion_sft.py`
  - 模仿固定权重 fusion
  - 在验证集上搜索最优 `alpha`
  - 在测试集上评测 `SASRec / SFT semantic / fixed fusion`

- `train_context_gate_sft.py`
  - 模仿 `3_gate_to_add_sasrec_score_into_LLM/train_gate_v2.py`
  - 先预计算 `SASRec score + SFT semantic score + prompt hidden states + sample stats`
  - 再训练 context-adaptive gate / dynamic fusion

辅助脚本：

- `precompute_sft_features.py`: 预计算 train/val/test 的融合特征 cache
- `manage_jobs.py`: 作为总控，自动找空闲 GPU，最多并发 4 张卡，启动并监控所有子进程

结果输出：

- `results/fixed_fusion_sft_results.json`
- `results/gate_default_results.json`
- `results/gate_hr1_results.json`
- `results/gate_topk_results.json`
- `jobs/manager_status.json`
- `logs/*.log`

运行方式：

```bash
cd temp_LLM_Agent_try/5_sft_sasrec_fusion
CUDA_VISIBLE_DEVICES=0,1,2,7 /home/kfwang/miniconda3/envs/onerec/bin/python manage_jobs.py
```
