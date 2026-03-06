#!/usr/bin/env python3
"""
Test script for item title generation using HuggingFace Transformers.
Run with: CUDA_VISIBLE_DEVICES=6,7 python test_title_generation_hf.py
"""

import os
import sys
import json
import subprocess as sp

print("=" * 60)
print("Testing Item Title Generation Pipeline (HF Transformers)")
print("=" * 60)

# Get script directory for path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))

metadata_path = os.path.join(script_dir, "../../Data/Amazons/data/meta_All_Beauty.jsonl")
mapping_path = os.path.join(script_dir, "../vanilla_sasrec/processed_data/item_mapping.json")
model_path = os.path.join(script_dir, "../../LLM4RecPart/models/Qwen3-1-7B")

# Check if data files exist
print("\nChecking data files...")

if not os.path.exists(metadata_path):
    print(f"✗ Metadata not found: {metadata_path}")
    sys.exit(1)
print(f"✓ Metadata found")

if not os.path.exists(mapping_path):
    print(f"✗ Item mapping not found: {mapping_path}")
    sys.exit(1)
print(f"✓ Item mapping found")

if not os.path.exists(model_path):
    print(f"✗ Model not found: {model_path}")
    sys.exit(1)
print(f"✓ Model found")

# Check GPU availability using nvidia-smi
print("\nChecking GPU availability...")
result = sp.run(["nvidia-smi", "-L"], capture_output=True, text=True)
if result.returncode != 0:
    print("✗ Cannot query GPUs with nvidia-smi")
    sys.exit(1)

num_physical_gpus = result.stdout.count("GPU ")
print(f"✓ {num_physical_gpus} physical GPUs detected")

# Check if GPUs 6 and 7 are available
for gpu_id in [6, 7]:
    if gpu_id >= num_physical_gpus:
        print(f"✗ GPU {gpu_id} not available (only {num_physical_gpus} physical GPUs)")
        sys.exit(1)
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

test_output_path = os.path.join(script_dir, "test_titles.json")

cmd = [
    sys.executable,  # Use current Python interpreter
    os.path.join(script_dir, "generate_item_titles_hf.py"),
    "--test_mode",
    "--batch_size", "4",
    "--output_path", test_output_path
]

print(f"\nCommand: {' '.join(cmd)}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print()

import subprocess
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
    print("")
    print("To run full inference on all 28,344 items:")
    print("  source ~/miniconda3/etc/profile.d/conda.sh && conda activate onerec")
    print("  CUDA_VISIBLE_DEVICES=6,7 python generate_item_titles_hf.py")
    print("=" * 60)
else:
    print(f"\n✗ Output file not created: {test_output_path}")
    sys.exit(1)
