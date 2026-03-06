# 7_retrieve_other_users_review_to_enhance_sft

这个目录把改动点从“推理时融合分数”转到“训练数据和输入 prompt 本身”。

核心思路：

- 对短序列用户，不再只给出其自身历史里的 review
- 额外从“该用户历史 item 的其他用户评论”里召回若干条 review
- 把这些跨用户 review 作为补充证据直接写进 prompt
- 训练时用增强后的 `train/val/test` 数据做 full SFT
- 推理时也对测试 prompt 做同样的 review retrieval 增强

当前默认设计：

- 短序列用户阈值：`history_len <= 4`
- 每个历史 item 最多补 `1` 条他人 review
- 每条样本最多补 `3` 条 review
- review 只从历史 item 中召回，不会碰 target item，避免泄漏

实验轨道：

- `base_init`
  - 从原始 `Qwen3-1-7B` 开始训练
- `strong_init`
  - 从 `4_finetune_to_upgrade_LLM/outputs/qwen3_title_sft/best_model` 继续训练

主要脚本：

- `prepare_augmented_sft_data.py`
  - 构造增强版 SFT 数据
- `train_augmented_sft.py`
  - 训练增强版 SFT 模型
- `evaluate_augmented_sft.py`
  - 在增强版测试 prompt 上评测
- `manage_jobs.py`
  - 用 `GPU 4/5/6/7` 调度训练与评测

运行：

```bash
cd temp_LLM_Agent_try/7_retrieve_other_users_review_to_enhance_sft
/home/kfwang/miniconda3/envs/onerec/bin/python manage_jobs.py
```

实验结果（2026-03-04）：

- 数据增强统计
  - 过滤后用户数：`4548`
  - 短序列用户数（`history_len <= 4`）：`3659`
  - `train` 平均额外 review 数：`0.823`
  - `val` 平均额外 review 数：`0.858`
  - `test` 平均额外 review 数：`1.266`
- `base_init`
  - `HR@1 = 0.1168`
  - `HR@10 = 0.1504`
  - `NDCG@10 = 0.1349`
- `strong_init`
  - `HR@1 = 0.1258`
  - `HR@10 = 0.1592`
  - `NDCG@10 = 0.1437`
- 对照 `4_finetune_to_upgrade_LLM/results/evaluation_metrics.json`
  - 原 corrected full SFT：`HR@1 = 0.1282`, `HR@10 = 0.1642`, `NDCG@10 = 0.1477`
  - 当前 `strong_init` 没有超过原 strong SFT
  - 差值：`HR@1 -0.24pt`, `HR@10 -0.51pt`, `NDCG@10 -0.40pt`

当前结论：

- “给短序列用户补充其他用户在其历史 item 上的 review” 这条方向可训练、可稳定收敛。
- 但在当前实现下，增强版 prompt 没有带来推荐指标提升，反而略低于原 `4_` 目录里的 strongest corrected full SFT。
- 更可能的问题不是训练失败，而是额外 review 引入了噪声，弱化了原本已经较强的 condensed-title next-item 信号。

关键结果文件：

- `data/dataset_summary.json`
- `results/base_init_evaluation_metrics.json`
- `results/strong_init_evaluation_metrics.json`
- `results/strong_init_sample_predictions.json`

---

保守版补跑（2026-03-05，已补完）：

- 使用 `experiment_config_conservative.yaml`
  - `short_history_threshold=2`
  - `max_aug_reviews_per_sample=1`
  - `max_aug_reviews_per_item=1`
  - `retrieved_review_max_len=80`
  - `recent_item_window=1`
  - `require_same_rating_bucket=true`
  - `min_review_chars=40`
  - `min_review_quality=45.0`
- 数据统计见 `data_conservative/dataset_summary.json`
  - `test` 平均额外 review 数：`0.278`（显著低于默认版 `1.266`）
- 最终指标（`results_conservative`）：
  - `base_init`: `HR@1=0.1269`, `HR@10=0.1566`, `NDCG@10=0.1428`
  - `strong_init`: `HR@1=0.1284`, `HR@10=0.1627`, `NDCG@10=0.1467`
- 与默认 7_ 结果相比（strong_init）：
  - `HR@1 +0.26pt`, `HR@10 +0.35pt`, `NDCG@10 +0.30pt`
- 与 `4_ corrected full SFT` 相比（`HR@1=0.1282, HR@10=0.1642, NDCG@10=0.1477`）：
  - `HR@1` 略高 `+0.02pt`
  - `HR@10/NDCG@10` 仍低 `-0.15pt/-0.10pt`

保守版关键文件：

- `data_conservative/dataset_summary.json`
- `results_conservative/base_init_evaluation_metrics.json`
- `results_conservative/strong_init_evaluation_metrics.json`
- `results_conservative/base_init_sample_predictions.json`
- `results_conservative/strong_init_sample_predictions.json`
