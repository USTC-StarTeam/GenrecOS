#!/usr/bin/env python3
"""
Generate condensed item titles using Qwen3-1-7B with vLLM.
Uses GPUs 6 and 7 for efficient batched inference.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

import torch
from vllm import LLM, SamplingParams


# =============================================================================
# Prompt Templates with Few-Shot Examples
# =============================================================================

TITLE_GENERATION_PROMPT = """You are a product title condenser. Given a product's full title, store name, and key details, generate a concise 2-4 word title that captures the essence of the product.

Here are some examples:

Example 1:
Full Title: Howard LC0008 Leather Conditioner, 8-Ounce (4-Pack)
Store: Howard Products
Condensed Title: Leather Conditioner

Example 2:
Full Title: Yes to Tomatoes Detoxifying Charcoal Cleanser (Pack of 2) with Charcoal Powder, Tomato Fruit Extract, and Gingko Biloba Leaf Extract, 5 fl. oz.
Store: Yes To
Condensed Title: Charcoal Cleanser

Example 3:
Full Title: Eye Patch Black Adult with Tie Band (6 Per Pack)
Store: Levine Health Products
Condensed Title: Eye Patch

Example 4:
Full Title: Neutrogena Hydro Boost Water Gel Moisturizer for Extra Dry Skin, 1.7 oz
Store: Neutrogena
Condensed Title: Hydro Boost Moisturizer

Example 5:
Full Title: Olay Regenerist Micro-Sculpting Cream, 1.7 oz
Store: Olay
Condensed Title: Regenerist Cream

Example 6:
Full Title: Dove Deep Moisture Body Wash, 24 fl oz
Store: Dove
Condensed Title: Deep Moisture Body Wash

Now generate a condensed title for:

Full Title: {title}
Store: {store}
Condensed Title:"""


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_metadata(metadata_path: str) -> Dict[str, Dict]:
    """Load item metadata from JSONL file."""
    logging.info(f"Loading metadata from {metadata_path}...")
    metadata = {}

    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading metadata"), 1):
            try:
                item_data = json.loads(line.strip())
                # Use parent_asin or asin as the key (try parent_asin first, then fall back)
                item_id = item_data.get('parent_asin') or item_data.get('asin')
                if item_id:
                    metadata[item_id] = item_data
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping line {line_num}: {e}")

    logging.info(f"Loaded {len(metadata)} items from metadata")
    return metadata


def load_item_mapping(mapping_path: str) -> Dict:
    """Load item mapping from JSON file."""
    logging.info(f"Loading item mapping from {mapping_path}...")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    logging.info(f"Item mapping: {mapping['num_items']} items")
    return mapping


def prepare_prompts(
    metadata: Dict[str, Dict],
    item_mapping: Dict
) -> List[str]:
    """
    Prepare prompts for all items in our vocabulary.
    Returns list of prompts and corresponding item IDs.
    """
    logging.info("Preparing prompts...")

    prompts = []
    item_ids = []

    # Get all original item IDs in our vocabulary
    for new_id in range(item_mapping['num_items']):
        # id_to_item maps new_id (as string) to original ASIN
        original_id = item_mapping['id_to_item'][str(new_id)]

        if original_id in metadata:
            item_data = metadata[original_id]
            title = item_data.get('title', 'Unknown Product')
            store = item_data.get('store', 'Unknown')

            # Format the prompt
            prompt = TITLE_GENERATION_PROMPT.format(title=title, store=store)
            prompts.append(prompt)
            item_ids.append(original_id)
        else:
            # Item not in metadata, use generic title
            prompts.append(f"Full Title: Unknown Product\nStore: Unknown\nCondensed Title: Unknown Product")
            item_ids.append(original_id)

    logging.info(f"Prepared {len(prompts)} prompts")
    return prompts, item_ids


# =============================================================================
# vLLM Inference
# =============================================================================

def generate_titles_with_vllm(
    prompts: List[str],
    model_path: str,
    gpu_ids: List[int],
    batch_size: int = 128,
    max_tokens: int = 20,
    temperature: float = 0.3,
    top_p: float = 0.9
) -> List[str]:
    """
    Generate condensed titles using vLLM.
    """
    # Set CUDA_VISIBLE_DEVICES to use only specified GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

    logging.info(f"Initializing vLLM on GPUs {gpu_ids}...")
    logging.info(f"Model: {model_path}")
    logging.info(f"Number of prompts: {len(prompts)}")
    logging.info(f"Batch size: {batch_size}")

    # Initialize LLM with vLLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=len(gpu_ids),  # Use multiple GPUs
        gpu_memory_utilization=0.85,  # Leave some headroom
        max_model_len=2048,
        trust_remote_code=True,
        disable_log_stats=True,
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["\n"],  # Stop at new line
        include_stop_str_in_output=False,
    )

    logging.info("Starting inference...")

    # Generate in batches
    all_outputs = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating titles"):
        batch_prompts = prompts[i:i + batch_size]

        outputs = llm.generate(batch_prompts, sampling_params)

        for output in outputs:
            # Get generated text (remove prompt)
            generated = output.outputs[0].text.strip()
            # Clean up: remove "Title:" prefix if present
            generated = generated.replace("Title:", "").strip()
            all_outputs.append(generated)

    logging.info(f"Generated {len(all_outputs)} titles")
    return all_outputs


# =============================================================================
# Post-processing
# =============================================================================

def postprocess_titles(generated_titles: List[str], item_ids: List[str]) -> List[Dict]:
    """
    Post-process generated titles.
    """
    results = []

    for item_id, title in zip(item_ids, generated_titles):
        # Clean up the title
        cleaned_title = title.strip()

        # Limit to max 6 words
        words = cleaned_title.split()
        if len(words) > 6:
            cleaned_title = ' '.join(words[:6])

        # Remove trailing punctuation
        cleaned_title = cleaned_title.rstrip('.,;:')

        results.append({
            'item_id': item_id,
            'condensed_title': cleaned_title
        })

    return results


# =============================================================================
# Main Function
# =============================================================================

def main():
    # Get script directory for path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))

    default_metadata = os.path.join(script_dir, "../../Data/Amazons/data/meta_All_Beauty.jsonl")
    default_mapping = os.path.join(script_dir, "../vanilla_sasrec/processed_data/item_mapping.json")
    default_model = os.path.join(script_dir, "../../LLM4RecPart/models/Qwen3-1-7B")
    default_output = os.path.join(script_dir, "item_titles.json")

    parser = argparse.ArgumentParser(description="Generate condensed item titles using Qwen with vLLM")
    parser.add_argument("--metadata_path", type=str, default=default_metadata,
                        help="Path to item metadata JSONL file")
    parser.add_argument("--mapping_path", type=str, default=default_mapping,
                        help="Path to item mapping JSON file")
    parser.add_argument("--model_path", type=str, default=default_model,
                        help="Path to Qwen model")
    parser.add_argument("--output_path", type=str, default=default_output,
                        help="Path to save generated titles")
    parser.add_argument("--gpu_ids", type=str, default="6,7",
                        help="GPU IDs to use (comma-separated)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for vLLM inference")
    parser.add_argument("--test_mode", action="store_true",
                        help="Test mode: only process first 100 items")

    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    logging.info(f"Using GPUs: {gpu_ids}")

    # Load data
    metadata = load_metadata(args.metadata_path)
    item_mapping = load_item_mapping(args.mapping_path)

    # Prepare prompts
    prompts, item_ids = prepare_prompts(metadata, item_mapping)

    # Test mode: limit to first 100 items
    if args.test_mode:
        logging.info("Test mode: limiting to first 100 items")
        prompts = prompts[:100]
        item_ids = item_ids[:100]

    # Generate titles with vLLM
    generated_titles = generate_titles_with_vllm(
        prompts=prompts,
        model_path=os.path.abspath(args.model_path),
        gpu_ids=gpu_ids,
        batch_size=args.batch_size,
        max_tokens=20,
        temperature=0.3,
        top_p=0.9
    )

    # Post-process
    results = postprocess_titles(generated_titles, item_ids)

    # Save results
    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved {len(results)} titles to {output_path}")

    # Print some examples
    logging.info("\nSample generated titles:")
    for i in range(min(5, len(results))):
        logging.info(f"  Item {results[i]['item_id']}: {results[i]['condensed_title']}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    main()
