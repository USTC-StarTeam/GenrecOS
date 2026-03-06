# Gate Network 实验：学习何时融合SASRec与LLM分数

## 2026-03-02 重构更新

这个目录里的旧版 `train_gate_v2.py` 已经证明存在明确问题，不能继续作为可信结论使用。当前版本已重构为 **context-adaptive full-catalog fusion**，并重新完成了一次全量训练与评估。

### 旧实现的关键问题

1. **训练/验证集 prompt 读取错误**
   - 旧版只读取了 `LLM_Rec_Data_Preparation/test.json`
   - `train/val` 没有真实 prompt，而是用占位符 `"Product"` 拼出来的假输入
   - 这会让 gate 训练看到的上下文特征与真实任务不一致

2. **LLM 生成切片错误**
   - 旧版在左侧 padding 的 batch 上，用每条样本自己的 `attention_mask.sum()` 去截取 `generate()` 输出
   - 但 continuation 实际上应从整批统一的 padded input length 开始
   - 结果是 prompt 本身被混进“预测文本”，导致后续 embedding similarity 严重污染

3. **模型结构过弱**
   - 旧版只学习一个 sample-level 标量 gate
   - 这个 gate 对所有 item 都使用同一个权重，无法表达 item-wise 的非线性融合

4. **训练目标与目标指标不匹配**
   - 旧版用 BPR + 均值负样本
   - 它更容易学成“整体偏向 SASRec”，而不是在全量候选集上真正提升 top-k 排名

### 新实现

当前 `train_gate_v2.py` 已重写为：

- 使用真实对齐后的 `train / val / test` prompt
- 批量预计算全量 item 上的：
  - SASRec 概率
  - LLM 生成文本 embedding 相似度
  - Qwen shallow / deep prompt hidden states
  - 样本级置信度统计特征
- 训练一个 **context-adaptive fusion model**
  - 以固定融合为 anchor
  - 学习 sample-specific 的非线性残差融合
  - 直接在全量 `28,344` 个 item 上做分类训练

### 最新结果

基于对齐测试集 `4,385` 样本，重算后的结果为：

| Method | HR@1 | HR@10 | NDCG@10 | 说明 |
|---|---:|---:|---:|---|
| SASRec | 9.97% | 14.00% | - | 顺序基线 |
| LLM semantic retrieval | 11.24% | 14.07% | - | 修复生成切片 bug 后显著回升 |
| Fixed Fusion (`alpha=0.55`) | **11.27%** | 15.62% | 0.1369 | 当前 HR@1 最优 |
| Context-Adaptive Fusion | 10.92% | **15.85%** | **0.1364** | 更好的深层 top-k 排名，但 HR@1 仍略低于固定融合 |

### 当前结论

- 旧版 gate 失败，**不是因为“动态融合完全没希望”**，而是因为实现里确实有严重 bug
- 修复后，LLM 语义通道恢复到了可用水平
- 当前最稳的结论是：
  - **固定融合** 仍然是最强 `HR@1`
  - **新的动态融合模型** 已经能在 `HR@10 / NDCG` 上与固定融合竞争，但还没有稳定超过固定融合的 `HR@1`
- 因此这个目录目前的可信结论应更新为：
  - `legacy gate result` 不可信
  - `fixed fusion after bug fix` 是当前最好的点估计
  - `context-adaptive fusion` 是可继续优化的下一步，而不是旧 README 里那种“完全学崩了”的状态

## 实验动机

前面的实验表明，Score Fusion（α=0.7）可以将SASRec和LLM的预测分数融合，达到HR@1=10.95%，超过了单一模型（SASRec: 9.22%, LLM: 9.26%）。

**核心问题**：能否训练一个Gate网络，动态地决定在哪些样本上更信任SASRec，哪些样本上更信任LLM？

## 方法设计

### 1. Gate网络结构

```
GateNetwork:
  输入 (hidden_state) → Linear(dim, 512) → LayerNorm → ReLU → Dropout(0.2)
                      → Linear(512, 256) → LayerNorm → ReLU → Dropout(0.2)
                      → Linear(256, 1) → Sigmoid()
  输出: gate_score ∈ [0, 1]
```

### 2. 三种Gate类型

| 类型 | 输入来源 | 输入维度 | 说明 |
|------|----------|----------|------|
| **Shallow** | Qwen第8层隐向量 | 2048 | 捕获浅层语义信息 |
| **Deep** | Qwen最终层隐向量 | 2048 | 捕获深层语义信息 |
| **Hybrid** | 拼接浅层+深层 | 4096 | 同时利用两层信息 |

### 3. 融合公式

```python
fused_score = gate_score * sasrec_prob + (1 - gate_score) * llm_similarity
```

- `gate_score ≈ 1`: 完全信任SASRec
- `gate_score ≈ 0`: 完全信任LLM
- `gate_score ≈ 0.5`: 两者平衡

### 4. Loss设计

```python
# BPR Loss: 让正样本的融合分数高于负样本
target_scores = fused[torch.arange(len(t)), t]  # 正样本分数
neg_scores = fused.mean(dim=1)  # 负样本平均分数
bpr_loss = -F.logsigmoid(target_scores - neg_scores).mean()

# 辅助MSE Loss: 让gate学习"谁分数高就用谁"
mse_loss = F.mse_loss(gate_scores, (sasrec_target > llm_target).float())

total_loss = bpr_loss + 0.1 * mse_loss
```

### 5. 训练策略

- **数据划分**: 训练集(train) / 验证集(val) / 测试集(test)，按时间顺序划分
- **预计算**: 提前计算所有SASRec分数、LLM隐状态、item embeddings，加速训练
- **Early Stopping**: 验证集HR@1连续10个epoch不提升则停止
- **优化器**: AdamW, lr=1e-3, weight_decay=1e-4

## 实验结果

### Baseline对比

| 方法 | Test HR@1 | Test HR@10 |
|------|-----------|------------|
| SASRec | 9.22% | 13.03% |
| LLM | 9.26% | 12.18% |
| **Score Fusion (α=0.7)** | **10.95%** | **13.25%** |

### Gate网络结果

| Gate类型 | Test HR@1 | Test HR@10 | Avg Gate | Std Gate |
|----------|-----------|------------|----------|----------|
| Shallow | 9.22% | 12.99% | 0.9991 | 0.0001 |
| Deep | 9.22% | 12.99% | 0.9995 | 0.000004 |
| Hybrid | 9.22% | 12.99% | 0.9994 | 0.000005 |

**关键发现**: 所有Gate类型都学到了输出接近1.0的值，即"永远信任SASRec"。

## 问题分析

### 问题1: Gate为什么趋向于1？

在训练数据上分析不同gate值的融合效果：

| Split | gate=0 (LLM) | gate=0.7 | gate=1 (SASRec) |
|-------|--------------|----------|-----------------|
| Train | 0.33% | 57.91% | **57.76%** |
| Val | 0.53% | **9.73%** | 9.51% |
| Test | 0.58% | **9.34%** | 9.22% |

**原因**: 在这个特征表示下，SASRec分数让target更容易排在前面，BPR loss自然会推动gate向1靠近。

### 问题2: LLM相似度计算有严重缺陷

我的实现：
```python
# 1. LLM生成预测文本
prediction = llm_model.generate(prompt)

# 2. 计算预测文本的embedding
pred_emb = llm_model(prediction).last_hidden_state.mean()

# 3. 与item embeddings计算cosine相似度
llm_similarity = cosine_similarity(pred_emb, item_embeddings)
```

**问题**：这种相似度计算的HR@1只有0.58%，而**真正的LLM（beam search生成）HR@1=9.26%**！

差距原因：
- 我的方法：embedding cosine相似度 → HR@1=0.58%
- 真正的LLM：beam search + prefix tree约束 → HR@1=9.26%

### 问题3: 训练集vs测试集的过拟合

| Split | SASRec HR@1 | 融合(gate=0.7) HR@1 |
|-------|-------------|---------------------|
| Train | **57.76%** | 57.91% |
| Test | 9.22% | 9.34% |

SASRec在训练集上过拟合严重（57.76% → 9.22%），但Gate网络只看到了训练集上的表现，因此学到了"永远相信SASRec"。

### 问题4: 为什么融合方法能超过SASRec？

LLM和SASRec的预测错误是不完全重叠的：

```
测试集4385个样本：
- SASRec正确: ~404个 (9.22%)
- LLM正确: ~406个 (9.26%)
- 重叠正确: ~330个
- SASRec独有: ~74个
- LLM独有: ~76个

融合后: 330 + 74 + 76 = ~480个 (10.95%)
```

融合的优势在于保留了两边的互补预测。

## 失败原因总结

1. **LLM特征表示错误**: 使用embedding相似度而非真实预测概率，严重低估了LLM的能力
2. **SASRec过拟合**: 训练集上SASRec表现太好，Gate学不到需要动态切换的场景
3. **BPR Loss的局限性**: 只关注相对排序，不关注是否真的需要动态切换

## 可能的改进方向

1. **使用LLM的真实预测概率**
   - 通过teacher forcing获取LLM对每个item的logits
   - 计算量大，但能反映LLM真实能力

2. **使用perplexity作为特征**
   - 高perplexity → LLM不确定 → 多用SASRec

3. **使用预测一致性作为特征**
   - SASRec和LLM一致时，置信度更高
   - 不一致时，需要动态选择

4. **改变训练目标**
   - 不用BPR loss，改用"融合后比单一模型提升"作为reward
   - 需要强化学习框架

## 文件说明

```
3_gate_to_add_sasrec_score_into_LLM/
├── train_gate.py          # 初版实现（有bug）
├── train_gate_v2.py       # 改进版实现（预计算特征）
├── train_gate.log         # 训练日志
├── cache/
│   └── precomputed_features.pkl  # 预计算的特征缓存
└── results/
    ├── gate_shallow_results.json
    ├── gate_deep_results.json
    ├── gate_hybrid_results.json
    └── gate_all_results.json
```

## 运行方式

```bash
cd temp_LLM_Agent_try/3_gate_to_add_sasrec_score_into_LLM
CUDA_VISIBLE_DEVICES=7 python train_gate_v2.py
```

## 结论

Gate网络实验未能成功学习到动态融合策略。核心问题是LLM特征的表示方式不正确，导致Gate网络无法感知LLM的真实预测能力。未来的工作需要使用LLM的真实预测概率作为特征，或者采用更复杂的训练框架（如强化学习）。
