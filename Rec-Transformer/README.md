# LLaMA-Rec: Simplified Transformer for Sequential Recommendation

A streamlined implementation of LLaMA-based recommendation model for sequential recommendation tasks, refactored from the original complex Fuxi-OneRec project for better maintainability and extensibility.

## 🎯 Overview

This repository contains a simplified and modularized version of LLaMA-Rec, originally designed for sequential recommendation. The model has been extracted from Hugging Face Transformers into an independent module (`llamarec/`) to enable easier customization and development.

### Key Features

- 🔧 **Independent Module**: LLaMA-Rec implementation decoupled from transformers source code
- 🎨 **Easy Customization**: Direct access to model internals for architectural modifications
- 📦 **Minimal Dependencies**: Uses only standard transformers APIs for maximum compatibility
- 🚀 **Ready-to-Train**: Complete training pipeline with configs for multiple datasets
- 🔄 **Backward Compatible**: Existing checkpoints work seamlessly with the new implementation

## 📁 Project Structure

```
Rec-Transformer-new/
├── llamarec/                    # 🏗️ Independent LLaMA-Rec implementation
│   ├── __init__.py             # Module exports
│   ├── configuration.py        # LlamaRecConfig class
│   ├── modeling.py             # Complete model implementation
│   ├── tokenization.py         # RQ code tokenizer
│   └── README.md              # Module documentation
├── try_train/                   # 🎯 Training scripts and configs
│   ├── train_single.py         # Single GPU training
│   ├── evaluate.py             # Model evaluation
│   ├── pretrain_config/        # YAML config files
│   └── run_train_single.sh     # Training launcher
├── data/                        # 📊 Dataset and processing
│   └── KuaiRand-27K-0501-Processed/
└── migration_example.py        # 📖 Usage examples
```

## 🚀 Quick Start

### Prerequisites

```bash
# Basic requirements
pip install torch transformers tokenizers datasets

# Optional for training
pip install tensorboard deepspeed
```

### Environment Setup

Add the project to your Python path:

```bash
cd /zhdd/home/kfwang/20250813Reproduct_Onerec/Fuxi-OneRec/Rec-Transformer-new
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# For permanent setup
echo 'export PYTHONPATH="${PYTHONPATH}:/zhdd/home/kfwang/20250813Reproduct_Onerec/Fuxi-OneRec/Rec-Transformer-new"' >> ~/.bashrc
source ~/.bashrc
```

### Basic Usage

```python
from llamarec import LlamaRecForCausalLM, LlamaRecConfig, create_rq_code_tokenizer

# Create model from config
config = LlamaRecConfig(
    vocab_size=10000,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
)
model = LlamaRecForCausalLM(config)

# Load from checkpoint
model = LlamaRecForCausalLM.from_pretrained("path/to/checkpoint")

# Create tokenizer for RQ codes
from datasets import Dataset
from llamarec import MockTrainingArguments
dataset = Dataset.from_dict({"text": ["<a_1> <b_2>", "<c_3> <d_4>"]})
args = MockTrainingArguments(output_dir="./tokenizer")
tokenizer = create_rq_code_tokenizer(dataset, args)
```

## 🎯 Training

### Single GPU Training

```bash
cd try_train
python train_single.py --config KuaiRand-27K-no-feature
```

### Configuration

Training configs are in `try_train/pretrain_config/`:

- `KuaiRand-27K-no-feature.yaml` - Main dataset config
- `KuaiRand-27K-100k.yaml` - Smaller subset for testing

### Key Training Parameters

```yaml
# Model architecture
hidden_size: 256
num_hidden_layers: 4
num_attention_heads: 4

# Training setup
per_device_train_batch_size: 8
learning_rate: 5e-5
num_train_epochs: 10
max_steps: 1000

# Data paths
data_path: "data/KuaiRand-27K-0501-Processed"
output_dir: "try_train/llama-rec-checkpoints"
```

## 📊 Datasets

### KuaiRand-27K (Primary)

- **Source**: Kuaishou recommendation dataset
- **Type**: Sequential recommendation with RQ codes
- **Size**: 100k sequences after processing
- **Features**: User-item interactions with temporal information

### Data Processing Pipeline

1. **Raw Data**: User interaction sequences with timestamps
2. **Preprocessing**: Generate positive train/test splits by timestamp
3. **ID Mapping**: Remap user/item IDs for model vocabulary
4. **RQ Encoding**: Convert items to Residual Quantization codes
5. **Format Conversion**: Transform to parquet for efficient training

## 🔧 Model Architecture

### LLaMA-Rec Components

- **`LlamaRecConfig`**: Model configuration with rec-specific parameters
- **`LlamaRecForCausalLM`**: Main model for causal language modeling
- **`LlamaRecAttention`**: Multi-head attention with RoPE
- **`LlamaRecMLP`**: Feed-forward layers with SwiGLU activation
- **`LlamaRecRMSNorm`**: RMS normalization layers

### Key Modifications from Standard LLaMA

- RQ code tokenization for items
- Sequential recommendation objective
- Custom training data collators
- Leave-one-out evaluation strategy

## 🎨 Customization

### Modifying Model Architecture

All model code is in `llamarec/modeling.py`. Example modifications:

```python
# Edit llamarec/modeling.py directly

class LlamaRecAttention(nn.Module):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Add custom attention modifications here
        # e.g., attention bias, alternative attention mechanisms
        ...

class LlamaRecMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Customize MLP layers
        # e.g., different activation functions, layer sizes
        ...
```

### Adding New Configuration Parameters

```python
# In llamarec/configuration.py
class LlamaRecConfig(PretrainedConfig):
    def __init__(self, my_custom_param=None, **kwargs):
        self.my_custom_param = my_custom_param
        super().__init__(**kwargs)

# Use in modeling.py
class LlamaRecForCausalLM(PreTrainedModel):
    def __init__(self, config):
        if config.my_custom_param:
            # Custom logic here
            pass
```

## 📈 Evaluation

### Metrics

- **Recall@K**: Top-K recall for recommendation
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate**: Binary hit rate evaluation

### Evaluation Scripts

```bash
# Standard evaluation
python evaluate.py --model_path path/to/checkpoint

# Beam search evaluation
python evaluate_beam.py --model_path path/to/checkpoint --beam_size 5
```

## 🔄 Migration from Original

This simplified version is migrated from the original complex implementation:

### What Changed

- ✅ **Extracted** `llamarec/` module from transformers source
- ✅ **Simplified** project structure and removed unused components
- ✅ **Streamlined** training pipeline with cleaner configs
- ✅ **Maintained** full backward compatibility with existing checkpoints

### Migration Benefits

- 🔧 **Easier Development**: Direct access to model code
- 📦 **Cleaner Dependencies**: No modified transformers required
- 🚀 **Faster Iteration**: Changes take effect immediately
- 🔄 **Better Maintenance**: Independent of transformers updates

## 🛠️ Development

### Testing New Changes

```python
# Run migration example to verify setup
python migration_example.py

# Quick model test
python -c "from llamarec import LlamaRecForCausalLM; print('✅ Import successful')"
```

### Code Structure

- **Configuration**: `llamarec/configuration.py`
- **Core Models**: `llamarec/modeling.py`
- **Tokenization**: `llamarec/tokenization.py`
- **Training**: `try_train/train_single.py`
- **Evaluation**: `try_train/evaluate.py`

## 📚 Documentation

- **Module Documentation**: `llamarec/README.md`
- **Migration Examples**: `migration_example.py`
- **Training Configs**: `try_train/pretrain_config/*.yaml`

## 🤝 Contributing

This is a research project. For model improvements:

1. Fork the repository
2. Modify `llamarec/` components as needed
3. Test with `migration_example.py`
4. Update configs in `try_train/pretrain_config/`
5. Submit pull request with clear description

## 📄 License

This project is for research purposes. Please check original dataset licenses for data usage.

## 🙏 Acknowledgments

- Original Fuxi-OneRec project team
- Hugging Face Transformers library
- KuaiRand dataset creators
- LLaMA model architecture

---

**Version**: 0.1.0 (Simplified)
**Last Updated**: October 2025
**Status**: Production Ready for Research