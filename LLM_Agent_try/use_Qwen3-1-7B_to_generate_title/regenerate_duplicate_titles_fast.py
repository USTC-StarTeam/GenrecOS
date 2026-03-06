#!/usr/bin/env python3
"""
Fast batch processing for regenerating unique titles for duplicate condensed titles.
Processes multiple duplicate groups in a single LLM call for speed.
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))


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


def create_batch_prompt(groups: List[Tuple[str, List[Dict]]]) -> str:
    """Create a batch prompt for multiple duplicate groups."""
    prompt_lines = ["Generate UNIQUE condensed titles (2-4 words) for these product groups:\n"]

    for group_idx, (current_title, items) in enumerate(groups, 1):
        prompt_lines.append(f"\n--- Group {group_idx} (current title: \"{current_title}\") ---")
        for i, item in enumerate(items[:3], 1):  # Limit to first 3 items per group for brevity
            full_title = item.get('full_title', 'Unknown')[:60]  # Truncate long titles
            prompt_lines.append(f"  {i}. {item['item_id']}: {full_title}")
        if len(items) > 3:
            prompt_lines.append(f"  ... and {len(items)-3} more items")

    prompt_lines.append("\nOutput format:")
    prompt_lines.append("Group N: title1, title2, ...")
    prompt_lines.append("\nGenerate all unique titles now:")

    return "\n".join(prompt_lines)


def generate_unique_titles_batch(
    model,
    tokenizer,
    groups: List[Tuple[str, List[Dict]]],
) -> Dict[str, str]:
    """Generate unique titles for a batch of duplicate groups."""

    if not groups:
        return {}

    prompt = create_batch_prompt(groups)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # Generate with shorter max tokens for speed
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.4,  # Lower temperature for more consistent output
            top_p=0.85,
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

    for group_idx, (current_title, items) in enumerate(groups):
        # Find the output for this group
        group_pattern = f"Group {group_idx + 1}:"
        if group_pattern in generated:
            # Find the line with this group's output
            lines = generated.split('\n')
            for line in lines:
                if line.strip().startswith(group_pattern):
                    titles_str = line.split(':', 1)[1].strip()
                    # Parse titles separated by commas
                    generated_titles = [t.strip().strip('.,;:') for t in titles_str.split(',')]
                    # Assign titles to items
                    for i, item in enumerate(items):
                        if i < len(generated_titles) and generated_titles[i]:
                            result[item['item_id']] = generated_titles[i][:8]  # Limit to 8 chars
                    break
            else:
                # If no proper line found, use fallback
                for i, item in enumerate(items):
                    full_title = item.get('full_title', 'Unknown')
                    words = full_title.split()[:3]
                    result[item['item_id']] = ' '.join(words)
        else:
            # Fallback: use first few words of full title
            for i, item in enumerate(items):
                full_title = item.get('full_title', 'Unknown')
                words = full_title.split()[:3]
                result[item['item_id']] = ' '.join(words)

    return result


def main():
    parser = argparse.ArgumentParser(description="Regenerate duplicate titles fast")
    parser.add_argument("--test_mode", action="store_true",
                        help="Only process first 5 batches")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Number of duplicate groups to process at once")
    parser.add_argument("--llm_batch_size", type=int, default=5,
                        help="Number of duplicate groups per LLM call")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Load data
    logging.info("Loading data...")
    titles, duplicates, metadata = load_data()

    logging.info(f"Total duplicate groups: {len(duplicates)}")

    # Filter to only meaningful duplicates
    meaningful_dups = [d for d in duplicates
                       if "Unknown" not in d['condensed_title']
                       and "____" not in d['condensed_title']
                       and "Product ID:" not in d['condensed_title']
                       and "The answer must" not in d['condensed_title']]

    logging.info(f"Meaningful duplicate groups to process: {len(meaningful_dups)}")

    if args.test_mode:
        meaningful_dups = meaningful_dups[:args.llm_batch_size * 2]
        logging.info(f"Test mode: processing {len(meaningful_dups)} groups")

    # Prepare groups for processing
    groups_to_process = []
    for dup in meaningful_dups:
        current_title = dup['condensed_title']
        items = dup['items']
        if len(items) > 1:
            # Add metadata to items
            for item in items:
                if item['item_id'] in metadata:
                    item['full_title'] = metadata[item['item_id']].get('title', 'Unknown')
                    item['store'] = metadata[item['item_id']].get('store', 'Unknown')
                else:
                    item['full_title'] = 'Unknown'
                    item['store'] = 'Unknown'
            groups_to_process.append((current_title, items))

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

    # Process in LLM batches
    all_updates = {}

    for i in tqdm(range(0, len(groups_to_process), args.llm_batch_size), desc="Processing batches"):
        batch_groups = groups_to_process[i:i + args.llm_batch_size]
        updates = generate_unique_titles_batch(model, tokenizer, batch_groups)
        all_updates.update(updates)

        # Add progress logging
        if (i // args.llm_batch_size + 1) % 10 == 0:
            logging.info(f"Processed {i + len(batch_groups)} groups ({len(all_updates)} updates)")

        if args.test_mode and i >= args.llm_batch_size * 2:
            break

    # Handle "Unknown" type duplicates by using first words of full title
    for dup_group in duplicates:
        if ("Unknown" in dup_group['condensed_title'] or
            "____" in dup_group['condensed_title'] or
            "Product ID:" in dup_group['condensed_title']):
            for item in dup_group['items']:
                if item['item_id'] in metadata:
                    full_title = metadata[item['item_id']].get('title', 'Unknown Product')
                else:
                    full_title = 'Unknown Product'
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
