#!/bin/bash

# Setup conda environment for GenrecOS
# This script creates a conda environment with all dependencies needed to run the training pipeline

set -e

ENV_NAME="GenrecOS"
PYTHON_VERSION="3.11" # Important!!!

echo "========================================"
echo "Setting up conda environment: ${ENV_NAME}"
echo "========================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing environment: ${ENV_NAME}"
    conda env remove -n ${ENV_NAME} -y
fi

# Create new conda environment
echo "Creating new conda environment with Python ${PYTHON_VERSION}..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}





pip install -r requirements_clean.txt

# Install PEFT for LoRA training
echo "Installing PEFT..."
pip install peft

# Install DeepSpeed for distributed training
echo "Installing DeepSpeed..."
pip install deepspeed

# Install vLLM for fast inference
echo "Installing vLLM..."
pip install vllm # when we install vllm, pytorch will be also installed


# Install utility packages
echo "Installing utility packages..."
pip install sentencepiece


# Verify installations
echo ""
echo "========================================"
echo "Verifying installations..."
echo "========================================"

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import peft; print(f'PEFT version: {peft.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"

echo ""
echo "========================================"
echo "Environment setup complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "    conda activate ${ENV_NAME}"
