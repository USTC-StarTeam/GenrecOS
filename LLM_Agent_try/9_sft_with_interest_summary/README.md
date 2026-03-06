# 9_sft_with_interest_summary

本实验在 `8_` 的两步兴趣总结思路上前移到 SFT 阶段：

1. 对 `train/val/test` 全部样本生成外部兴趣总结  
2. 总结输入包含该样本 history 的 review 文本与 item metadata  
3. 用增强后的 `train/val` 做全参数 SFT  
4. 在 `test(4548)` 与公共 `test_common_4385` 上评测

## 运行

```bash
cd temp_LLM_Agent_try/9_sft_with_interest_summary
/home/kfwang/miniconda3/envs/onerec/bin/python manage_jobs.py --allowed_gpus 0,1,2,3
```

## 产物

- 数据与摘要缓存：`data/`, `cache/`
- 训练输出：`outputs/qwen3_interest_sft/`
- 日志：`logs/`
- 评测结果：`results/*_metrics.json`
- 汇总：`results/summary.json`
- 调度状态：`jobs/manager_status.json`
