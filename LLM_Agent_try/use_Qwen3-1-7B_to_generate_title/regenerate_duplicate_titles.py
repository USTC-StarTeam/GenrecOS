#!/usr/bin/env python3
"""
Regenerate unique titles for items with duplicate condensed titles.
Optimized for faster processing with batched inference.
"""

import os
import json
import logging
import argparse
from typing import List, Dict
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Simplified prompt for faster generation
SIMPLIFIED_PROMPT = """For these {count} products that share the title "{title}", generate UNIQUE condensed titles (2-4 words each).

{items_text}

Output {count} unique titles, one per line:
1."""


def load_data():
    """Load all necessary data files."""
    # Load generated titles
    with open(os.path.join(script_dir, 'item_titles.json'), 'r') as f:
        titles = json.load(f)

    # Load duplicates info
    with open(os.path.join(script_dir, 'duplicate_titles.json'), 'r') as f:
        duplicates = json.load(f)

    # Load metadata
    metadata = {}
    metadata_path = os.path.join(script_dir, '../../Data/Amazons/data/meta_All_Beauty.jsonl')
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            item_data = json.loads(line.strip())
            item_id = item_data.get('parent_asin') or item_data.get('asin')
            if item_id:
                metadata[item_id] = item_data

    return titles, duplicates, metadata


def create_unique_titles_prompt(items: List[Dict], current_title: str) -> str:
    """Create a prompt to generate unique titles for a group of similar items."""

    items_info = []
    for i, item in enumerate(items, 1):
        full_title = item.get('full_title', 'Unknown')
        store = item.get('store', 'Unknown')
        rating = item.get('average_rating', 0)
        rating_count = item.get('rating_number', 0)

        info = f"""Item {i}:
  - ID: {item['item_id']}
  - Full Title: {full_title}
  - Store: {store}
  - Rating: {rating} ({rating_count} reviews)"""
        items_info.append(info)

    items_text = "\n\n".join(items_info)

    prompt = f"""You are a product title expert. The following {len(items)} products were given the same condensed title "{current_title}", but they are actually different products.

Please generate a UNIQUE and DISTINCTIVE condensed title (2-4 words) for EACH product based on their specific characteristics.

{items_text}

IMPORTANT RULES:
1. Each title must be UNIQUE - no two products should have the same title
2. Focus on the distinguishing features of each product
3. Consider the brand/store name if helpful
4. Keep titles concise (2-4 words)
5. If products are very similar, add differentiating details like size, color, or variant

Output format (one per line, exactly {len(items)} lines):
Item 1: [unique title]
Item 2: [unique title]
...and so on

Generate unique titles now:"""

    return prompt


def generate_unique_titles_batch(
    model,
    tokenizer,
    items: List[Dict],
    current_title: str
) -> Dict[str, str]:
    """Generate unique titles for a batch of similar items."""

    if len(items) <= 1:
        return {}

    # Skip items with "Unknown" type titles (these need different handling)
    if "Unknown" in current_title or "____" in current_title or "Product ID:" in current_title:
        # For these, just use the first few words of the full title
        result = {}
        for item in items:
            full_title = item.get('full_title', 'Unknown Product')
            words = full_title.split()[:4]
            result[item['item_id']] = ' '.join(words)
        return result

    prompt = create_unique_titles_prompt(items, current_title)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the generated part
    if prompt in generated:
        generated = generated[len(prompt):].strip()

    # Parse the output
    result = {}
    lines = generated.split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith('Item ') and ':' in line:
            try:
                # Parse "Item N: title"
                parts = line.split(':', 1)
                if len(parts) == 2:
                    item_num = int(parts[0].replace('Item ', '').strip())
                    title = parts[1].strip()
                    if 1 <= item_num <= len(items):
                        item_id = items[item_num - 1]['item_id']
                        # Clean up title
                        title = title.split('\n')[0].strip()  # Take first line
                        words = title.split()
                        if len(words) > 6:
                            title = ' '.join(words[:6])
                        result[item_id] = title
            except (ValueError, IndexError):
                continue

    # If we didn't get enough titles, fall back to original with suffix
    for i, item in enumerate(items):
        if item['item_id'] not in result:
            # Use first few words of full title as fallback
            full_title = item.get('full_title', 'Unknown')
            words = full_title.split()[:3]
            result[item['item_id']] = ' '.join(words)

    return result


def main():
    parser = argparse.ArgumentParser(description="Regenerate duplicate titles")
    parser.add_argument("--test_mode", action="store_true",
                        help="Only process first 10 duplicate groups")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of duplicate groups to process at once")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Load data
    logging.info("Loading data...")
    titles, duplicates, metadata = load_data()

    # Create title lookup
    title_lookup = {t['item_id']: t['condensed_title'] for t in titles}

    logging.info(f"Total duplicate groups: {len(duplicates)}")

    # Filter to only meaningful duplicates (skip "Unknown" types for special handling)
    meaningful_dups = [d for d in duplicates
                       if "Unknown" not in d['condensed_title']
                       and "____" not in d['condensed_title']
                       and "Product ID:" not in d['condensed_title']
                       and "The answer must" not in d['condensed_title']]

    logging.info(f"Meaningful duplicate groups to process: {len(meaningful_dups)}")

    if args.test_mode:
        meaningful_dups = meaningful_dups[:10]
        logging.info(f"Test mode: processing {len(meaningful_dups)} groups")

    # Load model
    logging.info("Loading model...")
    model_path = os.path.join(script_dir, '../../LLM4RecPart/models/Qwen3-1-7B')

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()

    logging.info("Model loaded successfully")

    # Process duplicates
    all_updates = {}

    for dup_group in tqdm(meaningful_dups, desc="Processing duplicates"):
        current_title = dup_group['condensed_title']
        items = dup_group['items']

        if len(items) <= 1:
            continue

        # Generate unique titles
        updates = generate_unique_titles_batch(model, tokenizer, items, current_title)
        all_updates.update(updates)

    # Handle "Unknown" type duplicates by using first words of full title
    for dup_group in duplicates:
        if ("Unknown" in dup_group['condensed_title'] or
            "____" in dup_group['condensed_title'] or
            "Product ID:" in dup_group['condensed_title']):
            for item in dup_group['items']:
                full_title = item.get('full_title', 'Unknown Product')
                words = full_title.split()[:4]
                all_updates[item['item_id']] = ' '.join(words)

    logging.info(f"Total updates to apply: {len(all_updates)}")

    # Apply updates to titles
    updated_count = 0
    for item in titles:
        if item['item_id'] in all_updates:
            new_title = all_updates[item['item_id']]
            if new_title and len(new_title) > 1:
                item['condensed_title'] = new_title
                updated_count += 1

    logging.info(f"Updated {updated_count} titles")

    # Save updated titles
    output_path = os.path.join(script_dir, 'item_titles_unique.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(titles, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved updated titles to {output_path}")

    # Print sample updates
    logging.info("\nSample updated titles:")
    sample_count = 0
    for item in titles:
        if item['item_id'] in all_updates and sample_count < 10:
            logging.info(f"  {item['item_id']}: {all_updates[item['item_id']]}")
            sample_count += 1


if __name__ == "__main__":
    main()
