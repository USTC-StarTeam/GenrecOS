#!/usr/bin/env python3
"""
Generate condensed item titles using Qwen3-1-7B with HuggingFace Transformers.
Uses GPUs 6 and 7 for efficient batched inference.
"""

import os
import json
import logging
import argparse
from typing import List, Dict
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# Prompt Templates with Few-Shot Examples
# =============================================================================

TITLE_GENERATION_PROMPT = """You are a product title condenser. Given a product's full title and store name, generate a concise 2-4 word title that captures the essence of the product.

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
                # Use parent_asin or asin as key
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
# HuggingFace Transformers Inference
# =============================================================================

def generate_titles_with_hf(
    prompts: List[str],
    model_path: str,
    batch_size: int = 32,
    max_tokens: int = 20,
    temperature: float = 0.3,
    top_p: float = 0.9
) -> List[str]:
    """
    Generate condensed titles using HuggingFace Transformers.
    Uses CUDA_VISIBLE_DEVICES environment variable for GPU selection.
    """
    # Determine device based on CUDA availability
    if torch.cuda.is_available():
        num_visible_gpus = torch.cuda.device_count()
        device = torch.device("cuda:0")  # Use first visible GPU
        logging.info(f"CUDA available, {num_visible_gpus} visible GPU(s)")
        logging.info(f"Using device: {device}")
        for i in range(num_visible_gpus):
            logging.info(f"  Visible GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available, using CPU")

    logging.info(f"Model: {model_path}")
    logging.info(f"Number of prompts: {len(prompts)}")
    logging.info(f"Batch size: {batch_size}")

    # Load tokenizer and model
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Set padding side to left for decoder-only models
    tokenizer.padding_side = 'left'

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"  # Let transformers handle device placement
    )
    model.eval()

    logging.info("Model loaded successfully")
    logging.info("Starting inference...")

    logging.info("Model loaded successfully")
    logging.info("Starting inference...")

    all_outputs = []

    # Generate in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating titles"):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract generated text (after the prompt)
        for prompt, output in zip(batch_prompts, batch_outputs):
            if output.startswith(prompt):
                generated = output[len(prompt):].strip()
            else:
                generated = output.strip()
            # Clean up
            generated = generated.split('\n')[0].strip()  # Take first line
            generated = generated.replace("Title:", "").strip()
            all_outputs.append(generated)

    logging.info(f"Generated {len(all_outputs)} titles")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return all_outputs


# =============================================================================
# Post-processing
# =============================================================================

def postprocess_titles(generated_titles: List[str], item_ids: List[str], metadata: Dict[str, Dict] = None) -> List[Dict]:
    """
    Post-process generated titles.
    """
    results = []

    for item_id, title in zip(item_ids, generated_titles):
        # Clean up the title
        cleaned_title = title.strip()

        # Take only first line (model might continue generating)
        if '\n' in cleaned_title:
            cleaned_title = cleaned_title.split('\n')[0].strip()

        # Remove common patterns that model might add
        patterns_to_remove = [
            "Wait, but",
            "So maybe",
            "The example",
            "Note:",
            "Example:",
        ]
        for pattern in patterns_to_remove:
            if pattern in cleaned_title:
                cleaned_title = cleaned_title.split(pattern)[0].strip()

        # Limit to max 6 words
        words = cleaned_title.split()
        if len(words) > 6:
            cleaned_title = ' '.join(words[:6])

        # Remove trailing punctuation
        cleaned_title = cleaned_title.rstrip('.,;:')

        # If title is too short or just symbols, try to use original title
        if len(cleaned_title) < 2 or all(c in '?_.,;:!' for c in cleaned_title.replace(' ', '')):
            if metadata and item_id in metadata:
                original_title = metadata[item_id].get('title', 'Unknown Product')
                # Use first few words of original title
                words = original_title.split()[:4]
                cleaned_title = ' '.join(words)
            else:
                cleaned_title = 'Unknown Product'

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

    parser = argparse.ArgumentParser(description="Generate condensed item titles using Qwen with HF Transformers")
    parser.add_argument("--metadata_path", type=str, default=default_metadata,
                        help="Path to item metadata JSONL file")
    parser.add_argument("--mapping_path", type=str, default=default_mapping,
                        help="Path to item mapping JSON file")
    parser.add_argument("--model_path", type=str, default=default_model,
                        help="Path to Qwen model")
    parser.add_argument("--output_path", type=str, default=default_output,
                        help="Path to save generated titles")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--test_mode", action="store_true",
                        help="Test mode: only process first 100 items")

    args = parser.parse_args()

    logging.info(f"Script directory: {script_dir}")
    logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

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

    # Generate titles with HF Transformers
    generated_titles = generate_titles_with_hf(
        prompts=prompts,
        model_path=os.path.abspath(args.model_path),
        batch_size=args.batch_size,
        max_tokens=20,
        temperature=0.3,
        top_p=0.9
    )

    # Post-process
    results = postprocess_titles(generated_titles, item_ids, metadata)

    # Save results
    output_path = os.path.abspath(args.output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if there's a parent directory
        os.makedirs(output_dir, exist_ok=True)

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
