# 8_interest_summary_before_generate

本实验验证两步推理范式：

1. 先用更强模型（`gpt-5-mini`）对用户历史进行兴趣总结  
2. 再把兴趣总结拼到推荐 prompt 中，让本地模型生成 next-item 语义，再做 embedding 检索评测

说明：若代理端点对 `gpt-5-mini` 返回空内容，脚本会自动回退到 `gpt-4o-mini`，避免写入无效占位摘要。

本目录默认统一到 `4385` 公共口径（与融合实验一致）：

- 输入测试集来自 `4_finetune_to_upgrade_LLM/data/test.jsonl`（`4548`）
- 通过 `vanilla_sasrec/processed_data/test.json + item_mapping.json` 对齐后得到 `4385`

## 运行

```bash
cd temp_LLM_Agent_try/8_interest_summary_before_generate
/home/kfwang/miniconda3/envs/onerec/bin/python manage_jobs.py --allowed_gpus 0,1,2,3
```

## 产物

- 数据与缓存：`data/`, `cache/`
- 日志：`logs/`
- 指标：`results/*_metrics.json`
- 汇总：`results/summary_common4385.json`
- 调度状态：`jobs/manager_status.json`
