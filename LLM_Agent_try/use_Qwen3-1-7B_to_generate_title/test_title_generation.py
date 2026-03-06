#!/usr/bin/env python3
"""
Test script for item title generation.
Run with --test_mode to verify the pipeline works on a small batch.
"""

import os
import subprocess
import sys
import json

# Setup conda environment
print("=" * 60)
print("Testing Item Title Generation Pipeline")
print("=" * 60)

# Check if conda env exists
result = subprocess.run(
    ["conda", "env", "list"],
    capture_output=True,
    text=True
)

if "onerec" not in result.stdout:
    print("Error: 'onerec' conda environment not found!")
    print("Available environments:")
    print(result.stdout)
    sys.exit(1)

print("\n✓ Conda environment 'onerec' found")

# Check if data files exist
print("\nChecking data files...")

# Get script directory for path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))

metadata_path = os.path.join(script_dir, "../../Data/Amazons/data/meta_All_Beauty.jsonl")
mapping_path = os.path.join(script_dir, "../vanilla_sasrec/processed_data/item_mapping.json")
model_path = os.path.join(script_dir, "../../LLM4RecPart/models/Qwen3-1-7B")

if not os.path.exists(metadata_path):
    print(f"✗ Metadata not found: {metadata_path}")
    sys.exit(1)
print(f"✓ Metadata found: {metadata_path}")

if not os.path.exists(mapping_path):
    print(f"✗ Item mapping not found: {mapping_path}")
    sys.exit(1)
print(f"✓ Item mapping found: {mapping_path}")

if not os.path.exists(model_path):
    print(f"✗ Model not found: {model_path}")
    sys.exit(1)
print(f"✓ Model found: {model_path}")

# Check vLLM installation
print("\nChecking vLLM installation...")
try:
    import vllm
    print(f"✓ vLLM installed (version {vllm.__version__})")
except ImportError:
    print("✗ vLLM not installed in current environment")
    print("Please install with: pip install vllm")
    sys.exit(1)

# Check GPU availability
print("\nChecking GPU availability...")
import torch
print(f"✓ PyTorch installed (version {torch.__version__})")

# Get actual GPU count (may be affected by CUDA_VISIBLE_DEVICES)
# We need to check the real physical GPUs by temporarily clearing CUDA_VISIBLE_DEVICES
import subprocess as sp
result = sp.run(["nvidia-smi", "-L"], capture_output=True, text=True)
if result.returncode != 0:
    print("✗ Cannot query GPUs with nvidia-smi")
    sys.exit(1)

num_physical_gpus = result.stdout.count("GPU ")
print(f"✓ CUDA available, {num_physical_gpus} physical GPUs detected")

# Check if GPUs 6 and 7 are available
required_gpus = [6, 7]
for gpu_id in required_gpus:
    if gpu_id >= num_physical_gpus:
        print(f"✗ GPU {gpu_id} not available (only {num_physical_gpus} physical GPUs)")
        sys.exit(1)
    # Get GPU name
    result = sp.run(["nvidia-smi", "-i", str(gpu_id), "--query-gpu=name", "--format=csv,noheader"],
                   capture_output=True, text=True)
    if result.returncode == 0:
        gpu_name = result.stdout.strip()
        print(f"✓ GPU {gpu_id}: {gpu_name}")
    else:
        print(f"✓ GPU {gpu_id}: Available")

# Run test mode (first 10 items)
print("\n" + "=" * 60)
print("Running Test Mode (10 items)")
print("=" * 60)

test_output_path = "./test_titles.json"

cmd = [
    "python", "generate_item_titles_vllm.py",
    "--test_mode",
    "--gpu_ids", "6,7",
    "--batch_size", "4",
    "--output_path", test_output_path
]

print(f"\nCommand: {' '.join(cmd)}")
print()

result = subprocess.run(cmd, capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

if result.returncode != 0:
    print(f"\n✗ Test failed with return code {result.returncode}")
    sys.exit(1)

# Check output
if os.path.exists(test_output_path):
    with open(test_output_path, 'r') as f:
        results = json.load(f)

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Generated {len(results)} titles")

    print("\nSample outputs:")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. Item {r['item_id']}: {r['condensed_title']}")

    # Clean up test output
    os.remove(test_output_path)
    print("\n✓ Test completed successfully!")
    print("\n" + "=" * 60)
    print("Pipeline is ready for full run!")
    print("Run: source ~/miniconda3/etc/profile.d/conda.sh && conda activate onerec")
    print("     CUDA_VISIBLE_DEVICES=6,7 python generate_item_titles_vllm.py --gpu_ids 6,7")
    print("=" * 60)
else:
    print(f"\n✗ Output file not created: {test_output_path}")
    sys.exit(1)
