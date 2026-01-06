# LLaMA-Rec 独立模块

## 概述

这是一个从 Hugging Face Transformers 库中解耦的 LLaMA-Rec 独立实现。将模型代码独立出来的主要目的是：

1. **便于深度修改**: 所有模型代码都暴露在这个目录中，可以自由修改模型结构
2. **独立维护**: 不再需要修改 transformers 源码
3. **可更新**: transformers 库可以随时更新到最新版本
4. **清晰架构**: 代码集中在一个模块中，结构清晰

## 模块结构

```
llamarec/
├── __init__.py           # 模块入口，导出公共 API
├── configuration.py      # LlamaRecConfig 配置类
├── modeling.py           # 所有模型实现（完整代码）
├── tokenization.py       # RQ code tokenizer
└── README.md            # 本文档
```

## 核心组件

### 1. 配置类

```python
from llamarec import LlamaRecConfig

config = LlamaRecConfig(
    vocab_size=10000,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    # ... 其他参数
)
```

### 2. 模型类

#### 主要使用: LlamaRecForCausalLM

```python
from llamarec import LlamaRecForCausalLM, LlamaRecConfig

# 方式1: 从配置创建
config = LlamaRecConfig(...)
model = LlamaRecForCausalLM(config)

# 方式2: 从检查点加载
model = LlamaRecForCausalLM.from_pretrained("path/to/checkpoint")
```

#### 其他模型变体

- `LlamaRecModel`: 基础模型（无任务头）
- `LlamaRecForSequenceClassification`: 序列分类
- `LlamaRecForQuestionAnswering`: 问答任务
- `LlamaRecForTokenClassification`: Token 分类

### 3. 模型组件（可单独修改）

所有内部组件都已导出，便于单独使用或继承修改：

```python
from llamarec import (
    LlamaRecRMSNorm,           # RMSNorm 层
    LlamaRecRotaryEmbedding,   # RoPE 位置编码
    LlamaRecMLP,               # MLP 层
    LlamaRecAttention,         # 注意力层
    LlamaRecDecoderLayer,      # 解码器层
)
```

### 4. Tokenizer

```python
from llamarec import create_rq_code_tokenizer, MockTrainingArguments
from datasets import Dataset

# 准备数据集
dataset = Dataset.from_dict({
    "text": ["<a_101> <b_42> <c_999>", "<b_42> <c_888>"]
})

# 创建 tokenizer
args = MockTrainingArguments(
    output_dir="./my_tokenizer",
    max_length=512
)
tokenizer = create_rq_code_tokenizer(dataset, args)
```

## 使用示例

### 完整训练流程

```python
from llamarec import LlamaRecForCausalLM, LlamaRecConfig, create_rq_code_tokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# 1. 创建或加载 tokenizer
tokenizer = create_rq_code_tokenizer(...)

# 2. 创建模型
config = LlamaRecConfig(
    vocab_size=len(tokenizer),
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
)
model = LlamaRecForCausalLM(config)

# 3. 加载数据
train_dataset = load_dataset(...)

# 4. 训练
training_args = TrainingArguments(...)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

### 从已有检查点加载

```python
from llamarec import LlamaRecForCausalLM

# 加载已有模型
model = LlamaRecForCausalLM.from_pretrained(
    "try_train/llama-rec-checkpoints/checkpoint-20"
)

# 模型已经可以使用
outputs = model.generate(...)
```

## 如何修改模型

所有模型代码都在 `modeling.py` 中，您可以直接修改：

### 示例1: 修改注意力机制

编辑 `llamarec/modeling.py`:

```python
class LlamaRecAttention(nn.Module):
    def forward(self, hidden_states, attention_mask, ...):
        # 在这里自由修改注意力计算逻辑
        # 例如：添加自定义的注意力偏置
        # 例如：使用不同的注意力计算方式
        ...
```

### 示例2: 修改 MLP 层

```python
class LlamaRecMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 修改 MLP 结构
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # 添加您自己的层
        self.custom_layer = nn.Linear(...)
        ...
```

### 示例3: 添加新的配置参数

编辑 `llamarec/configuration.py`:

```python
class LlamaRecConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=32000,
        # ... 现有参数
        my_custom_param=None,  # 添加新参数
        **kwargs,
    ):
        self.my_custom_param = my_custom_param
        # ...
```

然后在 `modeling.py` 中使用：

```python
class LlamaRecForCausalLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 使用新参数
        if config.my_custom_param:
            # 自定义逻辑
            ...
```

## 与原实现的兼容性

### 检查点兼容性

✅ **完全兼容**: 新模块使用相同的 `model_type="llama-rec"`，可以直接加载原有检查点。

### API 兼容性

✅ **完全兼容**: 所有公共方法签名保持不变，训练脚本只需修改导入路径。

## 迁移指南

### 更新训练脚本

**旧的导入** (嵌入在 transformers 中):
```python
from transformers.models.llama_rec.modeling_llamarec import LlamaRecForCausalLM
from transformers.models.llama_rec.configuration_llamarec import LlamaRecConfig
from transformers.models.llama_rec.tokenization_llamarec import create_rq_code_tokenizer
```

**新的导入** (独立模块):
```python
from llamarec import LlamaRecForCausalLM, LlamaRecConfig, create_rq_code_tokenizer
```

### 更新步骤

1. **确保 llamarec 在 Python 路径中**:
   ```bash
   cd /zhdd/home/kfwang/20250813Reproduct_Onerec/Fuxi-OneRec/Rec-Transformer
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **修改导入语句**: 使用上面的新导入方式

3. **验证**: 运行一个小测试确保模型可以正常加载

4. **可选**: 清理旧的 transformers 中的 llama_rec 目录

## 依赖关系

### 必需依赖

- `torch >= 2.0.0`
- `transformers >= 4.40.0` (标准版本，不需要修改)
- `tokenizers >= 0.13.0`

### 可选依赖

- `datasets`: 用于数据加载
- `tensorboard`: 用于训练可视化

## 常见问题

### Q: 如何确保使用的是新模块而不是旧的 transformers 版本？

A: 在脚本开头添加：
```python
import llamarec
print(llamarec.__file__)  # 应该输出 .../Rec-Transformer/llamarec/__init__.py
```
