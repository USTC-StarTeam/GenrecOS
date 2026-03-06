# 6_tool_call_to_add_sasrec_score_into_LLM

这个目录只做一件事：把“什么时候该引入 `SASRec score`”从连续 gate 改成离散工具调用。

核心思路：

- 给 tokenizer 新增单个 token：`[tool:seqscore]`
- 用 `LLM/SFT` 和 `SASRec` 的逐样本正确性比较来打伪标签
- 当 `LLM` top-1 错、但 `SASRec` top-1 对时，在训练答案前加上 `"[tool:seqscore]"`
- 训练后的模型在推理时如果自己输出了这个 token，就触发一次 `SASRec + semantic score` 融合；如果没输出，就只用语义匹配分数

这里同时跑两条实验线：

- `pre_sft`
  - teacher: 原始 base Qwen
  - init checkpoint: base Qwen
- `post_sft`
  - teacher: `4_finetune_to_upgrade_LLM/outputs/qwen3_title_sft/best_model`
  - init checkpoint: 同一个 best full SFT

主要脚本：

- `prepare_tool_sft_data.py`
  - 在对齐后的 `train/val/test` 上生成 teacher 预测
  - 根据 `tool_label = (llm_top1_wrong and sasrec_top1_correct)` 构造 tool-SFT 数据
- `train_tool_sft.py`
  - 扩 tokenizer
  - 训练会输出 `[tool:seqscore]` 的模型
- `evaluate_tool_fusion.py`
  - 检测生成结果里是否出现 tool token
  - 若出现，先去掉 token 再取 embedding，并把 `SASRec score` 融进来
- `manage_jobs.py`
  - 只在 `GPU 4/5/6/7`
  - 并行调度两条 teacher、两条训练、两条评测

输出目录：

- `data/pre_sft/`, `data/post_sft/`
- `outputs/pre_sft/`, `outputs/post_sft/`
- `results/pre_sft_tool_eval.json`, `results/post_sft_tool_eval.json`
- `jobs/manager_status.json`
- `logs/*.log`

运行方式：

```bash
cd temp_LLM_Agent_try/6_tool_call_to_add_sasrec_score_into_LLM
/home/kfwang/miniconda3/envs/onerec/bin/python manage_jobs.py
```
