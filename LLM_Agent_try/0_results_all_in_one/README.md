# temp_LLM_Agent_try 全实验结果总览

本目录用于把 `temp_LLM_Agent_try/` 里分散在各子文件夹中的实验，整理成一条完整实验线。这里不只记录“最好结果”，也强调每一阶段到底回答了什么问题、结果是否可比、哪些结论已经站稳、哪些还只是中间现象。

## 先看结论

当前这条实验线最重要的结论有十条：

1. **标题压缩是关键前处理**。把原始商品标题压成 2-4 词浓缩标题后，LLM zero-shot 推荐从 `HR@1=5.13% / HR@10=7.65%` 提升到 `9.26% / 12.18%`。
2. **固定权重分数融合是稳定有效的**。早期实验先证明了 `alpha=0.7` 可行；在 `3_gate_to_add_sasrec_score_into_LLM/` 修复数据与生成切片 bug 后，重新在对齐后的 `4385` 测试样本上计算，`Fixed Fusion (alpha=0.55)` 达到 `HR@1=11.27% / HR@10=15.62% / NDCG@10=13.70%`。
3. **“把 SASRec 候选直接塞进 prompt” 这条路基本失败**。Prompt Augmentation 最好也只有 `HR@1=5.56% / HR@10=7.73%`。
4. **在 `4385` 公共测试集上，把 SFT 再和 SASRec 做固定融合之后，效果还能继续提升。** `5_sft_sasrec_fusion/` 中，`SFT + SASRec fixed fusion (alpha=0.05)` 达到 `HR@1=14.69% / HR@10=17.17% / NDCG@10=15.99%`，超过单独 SFT。
5. **把“是否引入 SASRec”改造成工具调用式离散决策，目前没有带来进一步收益。** `6_tool_call_to_add_sasrec_score_into_LLM/` 中，两条 `tool-call routing` 轨道都出现了明显的过度触发 `[tool:seqscore]`，最终没有超过 `SFT + SASRec fixed fusion`。
6. **7_ 的 review-retrieval 增强在 `4385` 公共口径上重算后，保守版明显优于 7_ 默认版，并在 `HR@1` 上略高于 corrected full SFT，但 `HR@10/NDCG@10` 仍略低。** `7_retrieve_other_users_review_to_enhance_sft/` 中，`conservative + strong_init` 达到 `HR@1=13.14% / HR@10=16.72% / NDCG@10=15.03%`（test=4385）。
7. **8_ 的“两步兴趣总结再拼 prompt”在公共 `4385` 口径上没有提升，且明显弱于现有主线。** 修复 API 调用后重跑（`gpt-5-mini` 空返回时回退 `gpt-4o-mini`），`strong_init + interest` 只有 `HR@1=0.02% / HR@10=0.18% / NDCG@10=0.09%`，低于同轨道 `strong_init + raw`（`0.11% / 0.25% / 0.16%`）。
8. **9_ 的“在 SFT 阶段直接注入兴趣总结（train/val/test 全注入）”也没有带来提升。** `9_sft_with_interest_summary/` 中，公共 `4385` 口径下 `interest` 为 `HR@1=12.84% / HR@10=16.47% / NDCG@10=14.77%`，低于同管线 `raw` 的 `13.11% / 16.76% / 15.06%`。
9. **10_ 的“oracle反馈重生成”在统一口径下没有净提升。** 使用 `4_` 的浓缩标题 SFT best model 并只测 `4385` 后，`HR@1` 仅有极小上升（`13.09% -> 13.11%`），但 `HR@10/NDCG@10` 下降（`16.76% -> 16.58%`，`15.04% -> 15.01%`）。
10. **11_ 的“Seq2Pat pattern memory prompt”暂未形成净收益。** 在 `4548` 与 `4385` 两个口径上都表现为 `HR@1` 小幅上升，但 `HR@10/NDCG@10` 小幅下降（例如 `4385` 上 `HR@10: 16.85% -> 16.51%`）。

## 可比性说明

这里最大的坑不是模型，而是**不同实验并不总在同一测试集上比较**。主要有四套口径：

- `5312` 测试样本：`vanilla_sasrec/`
- `5397` 测试样本：`LLM_baseline_Prediction/`
- `4548` 测试样本：`LLM_Rec_Data_Preparation/` 和 `4_finetune_to_upgrade_LLM/`
- `4385` 测试样本：`1_vanilla_LLM_sasrec_combine/`、`2_find_time.../`、`3_gate.../` 的对齐子集

所以这份总结里，结果表会**按实验阶段和可比测试集分组**，而不会把所有数字混成一张“总榜”。

### 为什么会有 `4385`

`4385` 不是额外造出来的新数据集，而是一个**公共对齐测试子集**：

- `vanilla_sasrec/processed_data/test.json` 有 `5312` 个测试样本
- `LLM_Rec_Data_Preparation/test.json` 与 `4_finetune_to_upgrade_LLM/data/test.jsonl` 有 `4548` 个测试样本
- 融合实验要求同一条样本必须同时满足：
  - 在 LLM/SFT 测试集里存在 `(user_id, target_item_id)`
  - 这个 `target_item_id` 能通过 `item_mapping.json` 映射到 SASRec 的内部 id
  - 映射后的 `(user_id, ground_truth_internal_id)` 也确实出现在 SASRec 测试集里

满足这三个条件后的交集大小就是 `4385`。  
因此：

- 从 `4548` 到 `4385`，少掉了 `163` 条 LLM/SFT 测试样本
- 这些样本不能参与融合，不是因为评测脚本随便删数据，而是因为它们在 SASRec 测试集里没有严格一一对应项
- 只要方法需要同时比较 `SASRec` 和 `LLM`，就必须退到这个公共交集上

这也是为什么我现在把 `4385` 视为“全方法统一横比”的最合理口径。

## 4385 公共测试集总表

下面这张表只保留**已经统一到同一个 `4385` 公共测试集**上的结果。它比前面的分阶段表更适合横向比较所有方法谁强谁弱。

| 方法 | 配置 | HR@1 | HR@10 | NDCG@10 |
|---|---|---:|---:|---:|
| SFT + SASRec fixed fusion | `alpha=0.05` | **14.69%** | **17.17%** | **15.99%** |
| Corrected Full SFT | recomputed | 13.09% | 16.85% | 15.12% |
| Feedback Regen Oracle (10_) | `title_sft, round0 baseline, common4385` | 13.09% | 16.76% | 15.04% |
| SFT Interest In-Train (9_) | `raw, common4385` | 13.11% | 16.76% | 15.06% |
| Feedback Regen Oracle (10_) | `title_sft, final after feedback, common4385` | 13.11% | 16.58% | 15.01% |
| Seq2Pat Pattern Memory Prompt (11_) | `pattern_memory, common4385` | 13.30% | 16.51% | 14.97% |
| Review Retrieval Augment (conservative) | `7_, strong_init, common4385` | 13.14% | 16.72% | 15.03% |
| SFT Interest In-Train (9_) | `interest, common4385` | 12.84% | 16.47% | 14.77% |
| Review Retrieval Augment (default) | `7_, strong_init, common4385` | 12.91% | 16.15% | 14.66% |
| Fixed Fusion | `alpha=0.55` | 11.27% | 15.62% | 13.70% |
| Tool-Call Routed Fusion | `post_sft teacher` | 12.54% | 16.37% | 14.52% |
| Tool-Call Routed Fusion | `pre_sft teacher` | 11.22% | 14.37% | 13.01% |
| LLM semantic retrieval | context pipeline baseline | 11.24% | 14.07% | 12.71% |
| Score Fusion (legacy) | `alpha=0.7` | 10.95% | 13.25% | 12.23% |
| Perplexity Adaptive | `alpha=0.7, sens=0.05` | 10.95% | 13.32% | 12.25% |
| Dynamic Fusion | context-adaptive | 10.92% | 15.85% | 13.64% |
| SASRec | context pipeline baseline | 9.97% | 14.00% | 12.03% |
| Early Gate | shallow / deep / hybrid | 9.22% | 12.99% | - |
| LLM zero-shot (original titles) | recomputed | 5.72% | 8.42% | 7.01% |
| Prompt Augment | `k=20` | 5.56% | 7.73% | 6.57% |
| Interest Summary Prompt (8_) | `strong_init + raw` | 0.11% | 0.25% | 0.16% |
| Interest Summary Prompt (8_) | `strong_init + interest` | 0.02% | 0.18% | 0.09% |
| Interest Summary Prompt (8_) | `base_init + raw` | 0.00% | 0.07% | 0.03% |
| Interest Summary Prompt (8_) | `base_init + interest` | 0.00% | 0.05% | 0.02% |

如果只看这张统一总表，结论是：

- `SFT + SASRec fixed fusion (alpha=0.05)` 是当前统一口径下最强的方法
- 这说明在强 SFT 基础上，SASRec 仍然有可利用的补充信息
- 旧的 `Dynamic Fusion` 在 `HR@10` 上有价值，但当前还不如新的 `SFT + SASRec fixed fusion`
- 新的 `tool-call routing` 方向目前没有打赢强 SFT，也没有打赢 `SFT + SASRec fixed fusion`
- `10_` 的 oracle 反馈重生成在统一口径下只带来极小 `HR@1` 增益，但 `HR@10/NDCG@10` 退化，未形成净提升
- `11_` 的 seq2pat memory prompt 在两个口径都表现为 `HR@1` 小涨、`HR@10/NDCG@10` 下滑，暂未超过强 SFT 基线
- `Prompt Augment`、`Early Gate`、原始标题 zero-shot、`8_` 两步兴趣总结增强、`9_` 的 SFT内兴趣注入都已经可以视为失败路线

## 实验脉络

### 0. 标题压缩准备

文件夹：`use_Qwen3-1-7B_to_generate_title/`

这一阶段不直接做推荐，而是为后续所有 LLM 实验提供统一 item title 表达。最终得到 `28344` 个唯一浓缩标题。  
初始生成里有大量重复标题，`duplicate_titles.json` 记录了典型碰撞，例如 `"Magnetic Eyelashes"` 一度对应 `61` 个商品。后续通过去重再生成得到 `item_titles_unique.json`，这一步直接决定了后面 LLM 侧的可学性。

结论：**浓缩标题不是装饰，是后续性能提升的前提。**

### 1. vanilla SASRec 基线

文件夹：`vanilla_sasrec/`

这是整个目录里最早的序列推荐基线。  
数据过滤后得到：

- 用户数：`5312`
- 物品数：`28344`
- train：`12744`
- val：`5312`
- test：`5312`

最终测试结果来自 `checkpoints/sasrec_beauty_20260226_055626/test_results.json`：

| 方法 | Test Size | HR@1 | HR@5 | HR@10 | HR@20 | NDCG@10 |
|------|-----------|------|------|-------|-------|---------|
| SASRec | 5312 | 9.22% | 12.27% | 13.03% | 13.84% | 11.16% |

结论：**SASRec 提供了一个稳定且不弱的序列模式基线。**

### 2. LLM zero-shot：原始标题 vs 浓缩标题

#### 2.1 原始标题 baseline

文件夹：`LLM_baseline_Prediction/`

结果来自 `baseline_results_optimized.json`：

| 方法 | Test Size | HR@1 | HR@5 | HR@10 | HR@20 | NDCG@10 |
|------|-----------|------|------|-------|-------|---------|
| LLM + 原始标题 | 5397 | 5.13% | 6.87% | 7.65% | 8.49% | 6.33% |

结论：**直接拿原始长标题喂 LLM 效果很差。**

#### 2.2 浓缩标题 zero-shot

文件夹：`LLM_Rec_Data_Preparation/`

这一步把 prompt 改成 `浓缩标题 + rating + 截断 review`，同时采用和序列推荐一致的时间划分。结果来自 `evaluation_results.json`：

| 方法 | Test Size | HR@1 | HR@5 | HR@10 | HR@20 | NDCG@10 |
|------|-----------|------|------|-------|-------|---------|
| LLM + 浓缩标题 | 4548 | 9.26% | 11.41% | 12.18% | 12.91% | 10.68% |

相对原始标题，提升为：

- `HR@1`: `5.13% -> 9.26%`
- `HR@10`: `7.65% -> 12.18%`

结论：**标题压缩让 LLM 从“明显不可用”提升到“接近 SASRec 基线”。**

## 4385 对齐子集上的融合实验

这组实验都依赖 `LLM` 和 `SASRec` 的 item id 对齐，所以测试集缩成了 `4385`。

### 3. Prompt Augmentation

文件夹：`1_vanilla_LLM_sasrec_combine/`

结果来自 `results/prompt_augment_results.json`：

| 方法 | 配置 | HR@1 | HR@10 | 结论 |
|------|------|------|-------|------|
| Prompt Augment | k=5 | 2.35% | 3.17% | 很差 |
| Prompt Augment | k=10 | 3.49% | 5.02% | 很差 |
| Prompt Augment | k=20 | 5.56% | 7.73% | 最好但仍失败 |

结论：**把 SASRec 候选显式塞进 prompt 会干扰生成，不是有效融合方式。**

### 4. 固定权重 Score Fusion

文件夹：`1_vanilla_LLM_sasrec_combine/`

结果来自 `results/score_fusion_results.json`：

| 方法 | 配置 | HR@1 | HR@5 | HR@10 | HR@20 | NDCG@10 |
|------|------|------|------|-------|-------|---------|
| Score Fusion | alpha=0.3 | 9.87% | 10.97% | 11.11% | 11.31% | 10.56% |
| Score Fusion | alpha=0.5 | 10.86% | 12.31% | 12.38% | 12.66% | 11.73% |
| Score Fusion | alpha=0.7 | 10.95% | 13.11% | 13.25% | 13.48% | 12.23% |

这里的对齐子集 baseline 是：

- SASRec：`HR@1=9.22% / HR@10=13.03%`
- LLM：`HR@1=9.26% / HR@10=12.18%`

结论：

- 最优是 `alpha=0.7`
- **HR@1** 从 `9.22%` 提到 `10.95%`
- **HR@10** 从 `13.03%` 提到 `13.25%`

这是当前目录里**最早被验证有效的融合策略**。  
但它不是最新、也不是当前最强的固定融合结果。后续在 `3_gate_to_add_sasrec_score_into_LLM/` 修复实现 bug 后，同类对齐子集上的 fixed fusion 被重新计算为：

| 方法 | Test Size | HR@1 | HR@5 | HR@10 | HR@20 | NDCG@10 |
|------|-----------|------|------|-------|-------|---------|
| Fixed Fusion (recomputed) | 4385 | 11.27% | 15.10% | 15.62% | 16.05% | 13.70% |

所以现在更推荐把这组重算结果视为 `temp_LLM_Agent_try` 中 **可信的固定融合口径**。

### 5. 根据 LLM 置信度自适应调 alpha

文件夹：`2_find_time_to_add_sasrec_score_into_LLM/`

结果来自 `results/perplexity_adaptive_results.json`。核心思路是：用 LLM 困惑度估计不确定性，再动态调节 SASRec 权重。

最有代表性的配置：

| 方法 | 配置 | HR@1 | HR@10 | 相比固定 alpha=0.7 |
|------|------|------|-------|--------------------|
| Perplexity Adaptive | alpha=0.7, sens=0.05 | 10.95% | 13.32% | HR@1 持平，HR@10 小升 |
| Perplexity Adaptive | alpha=0.7, sens=0.2 | 10.90% | 13.36% | HR@1 略降，HR@10 小升 |

结论：**这条路只有边际收益，没有形成新的稳定优势。**  
它说明“什么时候更信 SASRec”这个问题是对的，但当前用 perplexity 做控制信号不够强。

## Gate / Adaptive Fusion 阶段

### 6. 早期 Gate Network：旧版结果不可信，且实测退化成 SASRec

文件夹：`3_gate_to_add_sasrec_score_into_LLM/`

最早的 `gate_shallow / gate_deep / gate_hybrid` 结果来自 `gate_all_results.json`：

| 方法 | Test HR@1 | Test HR@10 | Avg Gate | 现象 |
|------|-----------|------------|----------|------|
| Gate Shallow | 9.22% | 12.99% | 0.9991 | 几乎全信 SASRec |
| Gate Deep | 9.22% | 12.99% | 0.9995 | 几乎全信 SASRec |
| Gate Hybrid | 9.22% | 12.99% | 0.9994 | 几乎全信 SASRec |

结论：**这版 gate 没学到动态融合，直接塌成“永远选 SASRec”。**  
而且后续代码审查已经确认，这一版还存在两个实现问题，所以这些旧结果不能再被当作最终结论：

- `train/val` 没有使用真实 LLM prompt
- batch generation 的输出切片方式错误，导致 prompt 被混进预测文本，污染 LLM similarity

README 里已经解释了原因，核心有两点：

- 作为 gate 输入的 LLM 相似度特征太弱，不能代表真实 LLM 能力
- 在训练视角下，SASRec 信号过强，loss 会自然把 gate 推到 1

### 7. 修复后的 Context-Adaptive Fusion：重新对齐并重算

文件夹：`3_gate_to_add_sasrec_score_into_LLM/`

结果来自 `results/context_adaptive_fusion_results.json`。  
这版不是前面 gate 的简单重复，而是修复了数据与生成 bug 后，重做了特征预计算和融合网络，所以它自己的 baseline 也变了：

| 方法 | Test Size | HR@1 | HR@10 | NDCG@10 |
|------|-----------|------|-------|---------|
| SASRec baseline | 4385 | 9.97% | 14.00% | 12.03% |
| LLM baseline | 4385 | 11.24% | 14.07% | 12.71% |
| Fixed fusion (`alpha=0.55`) | 4385 | **11.27%** | 15.62% | **13.70%** |
| Dynamic fusion | 4385 | 10.92% | **15.85%** | 13.64% |

结论要分开看：

- 如果看 **HR@10**，dynamic fusion 比 fixed fusion 略高：`15.62% -> 15.85%`
- 如果看 **HR@1**，dynamic fusion 反而退步：`11.27% -> 10.92%`
- 如果看 **NDCG@10**，fixed fusion 仍略好：`13.70% > 13.64%`

所以这版结论不是“动态融合成功了”，而是：

- **旧版 gate 失败的很大一部分原因来自实现 bug，不是研究问题本身错误**
- **修复后，固定融合明显变强**
- **动态融合在 recall 层面有一点点价值，但 top-1 决策并没有超过 fixed fusion**

## 全参数 SFT 阶段

### 8. Qwen 全参数微调

文件夹：`4_finetune_to_upgrade_LLM/`

这是目前最新、也最完整的一条线。  
一开始我做过一版错误数据构建的 SFT，结果只有：

- `HR@1=8.30%`
- `HR@10=10.86%`

之后按照 `LLM_Rec_Data_Preparation/prepare_llm_rec_data.py` 的逻辑重构数据，再做全参数 SFT，最终结果来自 `results/evaluation_metrics.json`：

| 方法 | Test Size | HR@1 | HR@5 | HR@10 | HR@20 | NDCG@10 | Exact Match |
|------|-----------|------|------|-------|-------|---------|-------------|
| Corrected Full SFT | 4548 | 12.82% | 16.09% | 16.42% | 17.06% | 14.77% | 15.74% |

训练信息：

- 最佳 checkpoint：`checkpoint-1600`
- 最佳 `eval_loss`: `3.2269`
- 训练在 `epoch 4.785` 左右 early stop

结论：**在当前目录里，这就是目前最强的单项结果。**  
而且它说明，前一版 SFT 不行，主要不是“微调没用”，而是**数据构建错了**。

### 9. 用 SFT 模型继续和 SASRec 融合

文件夹：`5_sft_sasrec_fusion/`

这一阶段不再使用 base Qwen，而是把 `4_finetune_to_upgrade_LLM/` 训练出来的 `best_model` 作为新的 LLM 通道，再和 `SASRec` 做两类融合：

- 固定权重 score fusion
- context-adaptive gate / dynamic fusion

这批实验全部在同一个 `4385` 公共对齐测试集上完成。共享前置 cache 已经落盘到：

- `cache/sft_best_full/item_embeddings.pt`
- `cache/sft_best_full/train_features.pt`
- `cache/sft_best_full/val_features.pt`
- `cache/sft_best_full/test_features.pt`

#### 9.1 SFT 语义分数 + SASRec 固定融合

结果来自 `5_sft_sasrec_fusion/results/fixed_fusion_sft_results.json`：

| 方法 | Test Size | HR@1 | HR@5 | HR@10 | HR@20 | NDCG@10 |
|------|-----------|------|------|-------|-------|---------|
| SASRec | 4385 | 9.97% | 13.30% | 14.00% | 14.82% | 12.03% |
| SFT semantic | 4385 | 13.11% | 16.28% | 16.78% | 17.56% | 15.05% |
| SFT + SASRec fixed fusion | 4385 | **14.69%** | **16.83%** | **17.17%** | **17.83%** | **15.99%** |

验证集自动搜索出的最优权重是：

- `alpha=0.05`（SASRec 权重）

这个结果非常关键，因为它说明：

- 单独 SFT 已经很强
- 但在强 SFT 基础上，加入少量 SASRec 仍然能继续提升
- 最优 `alpha` 很小，说明这里应当“主要信 SFT，少量借 SASRec 修正”

#### 9.2 SFT + SASRec 的 context-adaptive gate

结果来自：

- `results/gate_default_results.json`
- `results/gate_hr1_results.json`
- `results/gate_topk_results.json`

三组配置的 baseline 都相同，都是上面的 `fixed fusion alpha=0.05`。  
动态 gate 的结果如下：

| 变体 | HR@1 | HR@5 | HR@10 | HR@20 | NDCG@10 | 现象 |
|------|------|------|-------|-------|---------|------|
| gate_default | 11.47% | 17.31% | **17.79%** | 18.31% | 15.03% | recall/top-k 更高，但 top-1 明显退化 |
| gate_hr1 | **11.70%** | 17.22% | 17.70% | 18.29% | **15.07%** | 三个 gate 里最偏 top-1，但仍远低于 fixed fusion |
| gate_topk | 11.38% | 17.29% | **17.79%** | **18.36%** | 14.98% | 更偏 recall，但 HR@1 最差 |

这三组 gate 有一个共同现象：

- `HR@10/20` 都比 fixed fusion 更高
- 但 `HR@1` 全部显著低于 fixed fusion 的 `14.69%`
- `NDCG@10` 也没有超过 fixed fusion 的 `15.99%`

因此这一阶段的结论不是“gate 成功超过 fixed fusion”，而是：

- **SFT + SASRec 的固定融合非常有效**
- **动态 gate 再次表现出“top-k 上升、top-1 退化”的老问题**
- **当前最可信、最强的新方案是 `SFT + SASRec fixed fusion (alpha=0.05)`**

### 10. 工具调用式离散路由：让模型自己决定何时引入 SASRec score

文件夹：`6_tool_call_to_add_sasrec_score_into_LLM/`

这一阶段不再训练连续 gate，而是尝试让模型通过输出特殊 token `[tool:seqscore]` 来决定是否调用 `SASRec score`。

具体做法是：

- 先扩 tokenizer，增加 `[tool:seqscore]`
- 先在对齐后的 `4385` 公共测试口径上，让 teacher 模型逐样本预测
- 若 `teacher LLM top-1 错` 且 `SASRec top-1 对`，就给该样本打上 `tool_label=true`
- SFT 训练时，在目标答案前加上 `[tool:seqscore]`
- 推理时若模型输出了这个 token，就先去掉它，再做 title embedding，并把 `SASRec score` 融进来

这一阶段同时跑了两条轨道：

- `pre_sft`
  - teacher: 原始 base Qwen
  - init checkpoint: 原始 base Qwen
- `post_sft`
  - teacher: `4_finetune_to_upgrade_LLM/outputs/qwen3_title_sft/best_model`
  - init checkpoint: 同一个 full SFT best model

#### 10.1 Teacher 打标签结果

结果来自：

- `6_tool_call_to_add_sasrec_score_into_LLM/data/pre_sft/teacher_summary.json`
- `6_tool_call_to_add_sasrec_score_into_LLM/data/post_sft/teacher_summary.json`

| 轨道 | train tool 比例 | val tool 比例 | test tool 比例 | teacher LLM(test) | SASRec(test) | 现象 |
|------|-----------------|---------------|----------------|-------------------|--------------|------|
| pre_sft | 92.70% | 10.39% | 9.74% | `HR@1=0.00% / HR@10=0.00%` | `9.76% / 13.89%` | base Qwen 作为 teacher 太弱，训练集几乎全被打成“应调用工具” |
| post_sft | 86.06% | 3.08% | 2.08% | `HR@1=12.79% / HR@10=16.90%` | `9.76% / 13.89%` | 强 SFT teacher 只在极少数样本上需要工具，但 train / test 分布仍然很不均衡 |

这里最关键的现象是：

- `pre_sft` 轨道里，teacher 几乎不会答对，所以训练集里绝大多数样本都被标成 `tool=true`
- `post_sft` 轨道里，teacher 已经很强，因此真正应当调用工具的样本在测试集里只有 `2.08%`
- 这意味着这个任务天然是一个**极端稀疏、极端不平衡的路由学习问题**

#### 10.2 Tool-SFT 训练

结果来自：

- `6_tool_call_to_add_sasrec_score_into_LLM/outputs/pre_sft/train_metrics.json`
- `6_tool_call_to_add_sasrec_score_into_LLM/outputs/post_sft/train_metrics.json`

| 轨道 | 训练停止 epoch | train loss | eval loss | 显存占用 |
|------|----------------|------------|-----------|----------|
| pre_sft | 3.125 | 3.4614 | 3.6112 | `alloc≈9.63GB, reserved≈30.46GB` |
| post_sft | 1.9231 | 2.9084 | 3.5214 | `alloc≈9.63GB, reserved≈30.46GB` |

这两条训练都顺利完成，没有 crash 或 OOM。  
但能否学到有效路由，最终还是要看评测时的**实际工具触发率**和**最终融合指标**。

#### 10.3 Tool-Call Routing 最终评测

结果来自：

- `6_tool_call_to_add_sasrec_score_into_LLM/results/pre_sft_tool_eval.json`
- `6_tool_call_to_add_sasrec_score_into_LLM/results/post_sft_tool_eval.json`

这里区分三种口径：

- `semantic_only`: 模型输出 title 后，只做 embedding 匹配
- `used_fixed_metrics`: 按验证集选出的固定 `alpha`，在所有样本上都融合
- `used_routed_metrics`: 只有模型输出 `[tool:seqscore]` 时才融合

| 轨道 | generated tool 比例 | teacher tool 比例 | semantic_only | used_fixed_metrics | used_routed_metrics | 结论 |
|------|---------------------|-------------------|---------------|--------------------|---------------------|------|
| pre_sft | 99.89% | 9.74% | `10.03% / 12.73% / 11.43%` | `11.22% / 14.37% / 13.01%` | `11.22% / 14.37% / 13.01%` | 几乎对所有样本都输出 tool token，已经退化成“总是融合” |
| post_sft | 84.17% | 2.08% | `12.52% / 15.51% / 14.06%` | `10.49% / 17.01% / 14.07%` | `12.54% / 16.37% / 14.52%` | top-1 比全量 fixed 更稳，但仍没有超过原始强 SFT，更没超过 `SFT + SASRec fixed fusion` |

说明：

- 表中三元组格式分别是 `HR@1 / HR@10 / NDCG@10`
- `post_sft` 的 routed 版本是这个目录里最合理的主结果，因为它确实在测试时执行了“检测 tool token 再决定是否融合”
- 但它的最终结果 `HR@1=12.54% / HR@10=16.37% / NDCG@10=14.52%`，仍低于：
  - 单独强 SFT 的统一口径结果 `13.09% / 16.85% / 15.12%`
  - `5_sft_sasrec_fusion/` 的固定融合 `14.69% / 17.17% / 15.99%`

这个阶段最值得记住的不是“又试了一个新花样”，而是它暴露出的失败模式：

- `pre_sft`：teacher 太弱，导致训练标签几乎全是正样本，模型学成“总是调用工具”
- `post_sft`：teacher 很强，真正需要路由的样本太稀疏，模型仍然大幅过触发，`generated_tool_rate=84.17%`，而 teacher 只希望它在 `2.08%` 的测试样本上调用工具

因此这一阶段的结论是：

- **工具调用式离散路由当前没有跑赢现有最强方案**
- **模型没有学会“何时少量调用工具”，而是明显过度调用**
- **这条路说明“显式学时机”这个方向值得探索，但当前伪标签定义和训练方式还不够好**

### 11. 基于“他人 review 召回”的 prompt 增强（7_）及保守版补跑

文件夹：`7_retrieve_other_users_review_to_enhance_sft/`

这一阶段尝试把增强点从“分数融合”改为“输入增强”：

- 对短历史样本，给 prompt 增加“其他用户在该历史 item 上的 review”
- 训练和推理都使用同构增强
- 同时跑 `base_init` 与 `strong_init` 两条轨道

#### 11.1 7_ 默认版（历史记录）

结果来自：

- `7_retrieve_other_users_review_to_enhance_sft/results/base_init_evaluation_metrics.json`
- `7_retrieve_other_users_review_to_enhance_sft/results/strong_init_evaluation_metrics.json`

| 轨道 | Test Size | HR@1 | HR@10 | NDCG@10 |
|------|-----------|------|-------|---------|
| base_init | 4548 | 11.68% | 15.04% | 13.49% |
| strong_init | 4548 | 12.58% | 15.92% | 14.37% |

默认版数据增强强度较高，`test` 平均额外 review 数为 `1.266`。

#### 11.2 7_ 保守版补跑（本次新增并补完）

结果来自：

- `7_retrieve_other_users_review_to_enhance_sft/results_conservative/base_init_evaluation_metrics.json`
- `7_retrieve_other_users_review_to_enhance_sft/results_conservative/strong_init_evaluation_metrics.json`
- `7_retrieve_other_users_review_to_enhance_sft/data_conservative/dataset_summary.json`

保守配置核心约束：

- `short_history_threshold=2`
- `max_aug_reviews_per_sample=1`
- `recent_item_window=1`
- `require_same_rating_bucket=true`
- `min_review_chars=40`
- `min_review_quality=45.0`

对应数据统计：

- `test` 平均额外 review 数降到 `0.278`（默认版是 `1.266`）

最终指标：

| 轨道 | Test Size | HR@1 | HR@10 | NDCG@10 |
|------|-----------|------|-------|---------|
| base_init (conservative) | 4548 | 12.69% | 15.66% | 14.28% |
| strong_init (conservative) | 4548 | 12.84% | 16.27% | 14.67% |

对比结论：

- 相比 7_ 默认版 `strong_init`：
  - `HR@1 +0.26pt`，`HR@10 +0.35pt`，`NDCG@10 +0.30pt`
- 相比 `4_finetune_to_upgrade_LLM` 的 corrected full SFT（`HR@1=12.82%, HR@10=16.42%, NDCG@10=14.77%`）：
  - `HR@1` 略高 `+0.02pt`
  - `HR@10/NDCG@10` 仍低 `-0.15pt/-0.10pt`

因此这条方向的当前判断是：

- “降低增强噪声”确实有效，能显著修复 7_ 默认版的退化
- 但在 `4548` 主口径上，还没有形成对 `4_ corrected full SFT` 的全面超越

#### 11.3 按你要求统一到 `4385` 公共口径后的重算（本次新增）

为保证与融合实验完全可比，额外把 7_ 的 `default/conservative` 两条轨道都重算到公共 `4385`：

- `7_/data/test_common_4385.jsonl`（默认增强）
- `7_/data_conservative/test_common_4385.jsonl`（保守增强）

两套数据都满足同一对齐规则：

- 输入 `4548` 条
- 输出 `4385` 条
- 丢弃 `163` 条未与 SASRec test 严格对齐样本

公共口径结果：

| 配置 | Test Size | HR@1 | HR@10 | NDCG@10 |
|------|-----------|------|-------|---------|
| 7_ default + base_init | 4385 | 12.09% | 15.44% | 13.88% |
| 7_ default + strong_init | 4385 | 12.91% | 16.15% | 14.66% |
| 7_ conservative + base_init | 4385 | 13.07% | 16.05% | 14.68% |
| 7_ conservative + strong_init | 4385 | **13.14%** | **16.72%** | **15.03%** |

与 `4385` 口径的 `Corrected Full SFT`（`13.09% / 16.85% / 15.12%`）比较：

- `7_ conservative + strong_init` 在 `HR@1` 上略高 `+0.05pt`
- 但 `HR@10/NDCG@10` 仍低 `-0.14pt/-0.09pt`

所以统一到公共口径后，结论依然是：

- 保守版确实比 7_ 默认版更好
- 但仍未在 `HR@10/NDCG@10` 上全面超过 corrected full SFT

### 12. 两步兴趣总结再拼接 prompt（8_，修复后重跑）

文件夹：`8_interest_summary_before_generate/`

这一阶段的设计是：

- 先调用更强模型做“用户兴趣总结”
- 把总结文本拼接回原始推荐 prompt
- 分别在 `base_init` 与 `strong_init` 上评测 `raw` vs `interest` 两种输入

#### 12.1 首轮失败原因与修复

首轮跑出来大量 `- no clear preference`，根因不是评测脚本，而是接口返回空内容：

- 在代理端点 `https://xiaoai.plus/v1/chat/completions` 上，`gpt-5-mini` 的 `message.content` 大量为空
- 原脚本把空内容写成固定占位文案，导致兴趣摘要几乎失真

修复后重跑策略：

- 保留 `gpt-5-mini` 作为主模型
- 当主模型返回空内容时自动回退 `gpt-4o-mini`
- 对无效缓存自动刷新重算

修复后摘要统计（`4548` 全量）：

- `primary_count=0`
- `fallback_count=4547`
- `unavailable_count=0`
- `error_count=1`（缓存中有 1 条 `unavailable summary`）

#### 12.2 公共 `4385` 口径结果

结果来自 `8_interest_summary_before_generate/results/summary_common4385.json`：

| 配置 | HR@1 | HR@10 | NDCG@10 |
|------|------|-------|---------|
| base_init + raw | 0.00% | 0.07% | 0.03% |
| base_init + interest | 0.00% | 0.05% | 0.02% |
| strong_init + raw | 0.11% | 0.25% | 0.16% |
| strong_init + interest | 0.02% | 0.18% | 0.09% |

结论：

- 在这条实现上，`interest` 相比 `raw` 仍是退化
- 即使修复摘要生成后，也远低于 `4_/5_/7_` 主线结果
- 当前可把 `8_` 归类为“接口已修复，但方法路径暂未成立”

### 13. 在 SFT 阶段直接引入兴趣总结（9_）

文件夹：`9_sft_with_interest_summary/`

这一阶段和 `8_` 的核心区别是：

- 不是只在推理阶段加兴趣总结，而是从 **SFT 数据构建开始**就把 summary 注入 train/val/test
- summary prompt 同时使用 `history + review + item metadata`
- 目标是让模型从训练期就适应这类外部总结输入

结果来自 `9_sft_with_interest_summary/results/summary.json`：

| 配置 | Test Size | HR@1 | HR@10 | NDCG@10 |
|------|-----------|------|-------|---------|
| raw_test4548 | 4548 | 12.77% | 16.45% | 14.71% |
| interest_test4548 | 4548 | 12.55% | 16.12% | 14.45% |
| raw_common4385 | 4385 | 13.11% | 16.76% | 15.06% |
| interest_common4385 | 4385 | 12.84% | 16.47% | 14.77% |

其中 `interest - raw` 的差值：

- `4548`: `HR@1 -0.22pt / HR@10 -0.33pt / NDCG@10 -0.26pt`
- `4385`: `HR@1 -0.27pt / HR@10 -0.30pt / NDCG@10 -0.28pt`

结论：

- 在这轮实现里，把兴趣总结前移到 SFT 阶段并没有提升指标
- 同一训练与评测管线下，`raw` 输入稳定优于 `interest` 输入

### 14. Oracle 反馈重生成验证（10_，统一口径）

文件夹：`10_feedback_regeneration_oracle/`

按统一要求重跑设置：

- checkpoint 改为 `4_finetune_to_upgrade_LLM/outputs/qwen3_title_sft/best_model`
- 仅使用 `raw_common4385`，不再评测 `4548`
- 不使用 interest 增强
- 生成规则：`round0` 先生成；若 target 不在 top-5 检索结果中，则反馈“上一轮错误，请重生成”，最多 3 轮反馈

结果来自 `10_feedback_regeneration_oracle/results/feedback_regen_title_sft_common4385_metrics.json`：

| 配置 | Test Size | HR@1 | HR@10 | NDCG@10 |
|------|-----------|------|-------|---------|
| round0 baseline | 4385 | 13.09% | 16.76% | 15.04% |
| final after feedback | 4385 | 13.11% | 16.58% | 15.01% |

delta (`final - baseline`)：

- `HR@1 +0.02pt`
- `HR@10 -0.18pt`
- `NDCG@10 -0.03pt`

诊断：

- feedback trigger rate: `83.76%`
- avg generation calls per sample: `3.51`

结论：

- 在 oracle 设定下，反馈重生成只带来极小 top-1 增益
- 但 top-10 与 ndcg 出现退化，整体未形成净提升

### 15. Seq2Pat Pattern Memory Prompt（11_）

文件夹：`11_seq2pat_memory_prompt/`

设计目标：

- 从用户历史序列中挖掘可复用的顺序 pattern 作为 memory
- 在训练/推理时对每条样本做子序列匹配，把匹配到的 pattern 拼接进 prompt 作为辅助信号
- 检验“显式行为模式记忆”是否能在强 SFT 基线之上继续带来增益

方法实现要点（按最终执行版本）：

- 使用 `sequential.seq2pat` 在训练可见历史（每个用户去掉最后一个测试交互）上挖掘全局 pattern
- 仅使用“子序列匹配”，不按用户ID筛选 memory
- prompt 中附加最多 `top-3` 条 pattern，含 `full/partial` 匹配标记与支持度

数据与 memory 统计（来自 `11_/results/summary.json`）：

- `mined_pattern_count = 332`
- `test pattern_matches_avg = 0.10`
- `test pattern_matches_nonzero_ratio = 5.91%`

结果：

| 配置 | Test Size | HR@1 | HR@10 | NDCG@10 |
|------|-----------|------|-------|---------|
| 11_ pattern memory | 4548 | 12.99% | 16.20% | 14.68% |
| 11_ pattern memory | 4385 | 13.30% | 16.51% | 14.97% |

对比基线：

- 对比 `4_` corrected full SFT（4548: `12.82% / 16.42% / 14.77%`）：
  - `HR@1 +0.18pt`
  - `HR@10 -0.22pt`
  - `NDCG@10 -0.09pt`
- 对比 `4385` 口径 corrected full SFT（`13.09% / 16.85% / 15.12%`）：
  - `HR@1 +0.21pt`
  - `HR@10 -0.34pt`
  - `NDCG@10 -0.15pt`

结论：

- 11_ 在两个口径都出现同一趋势：top-1 小幅改善，但 top-10 与 ndcg 下降
- 当前不构成对强 SFT 主线的净收益提升

## 汇总表

### A. 各阶段代表结果

| 阶段 | 文件夹 | 代表结果 | 结论 |
|------|--------|----------|------|
| 标题压缩 | `use_Qwen3-1-7B_to_generate_title/` | 生成 `28344` 个唯一浓缩标题 | 后续 LLM 实验的基础 |
| SASRec 基线 | `vanilla_sasrec/` | `HR@1=9.22% / HR@10=13.03%` | 稳定序列基线 |
| LLM 原始标题 | `LLM_baseline_Prediction/` | `5.13% / 7.65%` | 明显不够好 |
| LLM 浓缩标题 | `LLM_Rec_Data_Preparation/` | `9.26% / 12.18%` | 标题压缩显著有效 |
| Prompt Augment | `1_vanilla_LLM_sasrec_combine/` | `5.56% / 7.73%` | 失败 |
| Score Fusion | `3_gate.../context_adaptive_fusion_results.json` | `11.27% / 15.62%` | 当前可信的最佳固定融合 |
| Perplexity Adaptive | `2_find_time.../` | `10.90% / 13.36%` | 仅小幅改 recall |
| Early Gate | `3_gate.../gate_all_results.json` | 约 `9.22% / 12.99%` | 旧版实现有 bug，结果不可信 |
| Context-Adaptive Fusion | `3_gate.../context_adaptive_fusion_results.json` | `10.92% / 15.85%` | recall 有增益，top-1 仍不如 fixed fusion |
| Corrected Full SFT | `4_finetune_to_upgrade_LLM/` | `12.82% / 16.42%` | 当前最强 |
| SFT + SASRec Fixed Fusion | `5_sft_sasrec_fusion/` | `14.69% / 17.17%` | 当前 `4385` 统一口径最强 |
| SFT + SASRec Dynamic Gate | `5_sft_sasrec_fusion/` | `11.38%~11.70% / 17.70%~17.79%` | top-k 有增益，top-1 明显退化 |
| Tool-Call Routed Fusion | `6_tool_call_to_add_sasrec_score_into_LLM/` | `12.54% / 16.37%` | 没有超过强 SFT，也没有超过固定融合 |
| Review Retrieval Augment (default) | `7_retrieve_other_users_review_to_enhance_sft/` | `12.91% / 16.15%` (`4385`, strong_init) | 公共口径下可训练，但弱于保守版 |
| Review Retrieval Augment (conservative) | `7_retrieve_other_users_review_to_enhance_sft/` | `13.14% / 16.72%` (`4385`, strong_init) | 公共口径下优于默认版，接近 corrected full SFT |
| Interest Summary Prompt (two-step) | `8_interest_summary_before_generate/` | `0.02% / 0.18%` (`4385`, strong_init+interest) | 接口修复后已重跑，但当前路径明显失败 |
| SFT Interest In-Train | `9_sft_with_interest_summary/` | `13.11% / 16.76%` (`4385`, raw) | 同管线对比下，interest 注入版退化到 `12.84% / 16.47%` |
| Feedback Regeneration Oracle | `10_feedback_regeneration_oracle/` | `13.11% / 16.58%` (`4385`, final) | 统一口径重跑后，HR@1仅+0.02pt，HR@10/NDCG@10下降，未形成净提升 |
| Seq2Pat Pattern Memory Prompt | `11_seq2pat_memory_prompt/` | `13.30% / 16.51%` (`4385`) | 相比强SFT，HR@1小升但HR@10/NDCG@10下降，未形成净收益 |

### B. 现在最稳的判断

- **已经被充分验证的有效改动**：
  - 浓缩标题
  - 固定权重分数融合
  - 正确数据构建后的全参数 SFT
  - 在强 SFT 基础上，再加入少量 SASRec 的固定融合

- **已经被证明效果不好的方向**：
  - 原始长标题直接 zero-shot
  - 把 SASRec 候选直接放进 prompt
  - 早期 gate 版本
  - 两步兴趣总结后再拼 prompt（8_）
  - 在 SFT 阶段直接注入兴趣总结（9_）
  - Seq2Pat pattern memory prompt（11_，当前版本）

- **还有探索价值，但证据不够硬的方向**：
- perplexity / confidence 驱动的自适应权重
- context-adaptive fusion
- SFT 基础上的 dynamic gate（当前 top-k 有用，但 top-1 退化）
- tool-call routing（当前显式调用 token 明显过触发，尚未体现出真正“择机调用”的能力）
- review retrieval 数据增强（公共 `4385` 口径下，保守版已显著优于默认版，但还未在 `HR@10/NDCG@10` 全面超越 corrected full SFT）
- oracle 反馈重生成（10_）在统一口径下只提升了 `HR@1`，但 `HR@10/NDCG@10` 下降，尚不能证明“反思机制”稳定有效

## 最后给出一句话总结

这整个 `temp_LLM_Agent_try/` 的主线非常清楚：

**先靠浓缩标题把 LLM 从不可用拉到可用，再尝试和 SASRec 融合，最后走向真正的 supervised tuning。到目前为止，效果最强的仍然不是花哨 gate 或 tool routing，而是“正确数据构建后的强 SFT，再加一个很轻的固定融合”。**
