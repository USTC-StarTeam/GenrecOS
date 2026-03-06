# Generate Condensed Item Titles with Qwen3-1-7B

This directory contains scripts for generating condensed (2-4 word) item titles from original product titles using the Qwen3-1-7B language model.

## Purpose

Convert long, verbose product titles into concise, meaningful short titles that are more suitable for LLM-based recommendation tasks.

## Output

The main output is `item_titles_unique.json` containing 28,344 unique condensed titles:

```json
{
  "B00XXXXXXXX": "Moisturizing Face Cream",
  "B01YYYYYYYY": "Organic Lip Balm",
  ...
}
```

## Files

```
use_Qwen3-1-7B_to_generate_title/
├── generate_item_titles_hf.py              # Main generation script (HuggingFace)
├── generate_item_titles_vllm.py            # Alternative using vLLM (faster)
├── test_title_generation.py                # Test vLLM setup
├── test_title_generation_hf.py             # Test HuggingFace setup
├── regenerate_duplicate_titles.py          # Fix duplicate titles (slow)
├── regenerate_duplicate_titles_fast.py     # Fix duplicate titles (fast)
├── item_titles.json                        # Initial output (with duplicates)
├── item_titles_unique.json                 # Final output (unique titles)
└── duplicate_titles.json                   # List of items with duplicate titles
```

## Pipeline

### Step 1: Generate Initial Titles

```bash
# Using HuggingFace Transformers
python generate_item_titles_hf.py

# Or using vLLM (faster, requires vLLM installation)
python generate_item_titles_vllm.py
```

This generates `item_titles.json` with condensed titles for all items.

### Step 2: Fix Duplicate Titles

Some items may receive identical condensed titles. The regeneration process ensures uniqueness:

```bash
# Fast version (recommended)
python regenerate_duplicate_titles_fast.py

# Slow version (fallback)
python regenerate_duplicate_titles.py
```

Output: `item_titles_unique.json` with unique titles for each item.

## Generation Prompt

The model is prompted to generate 2-4 word summaries:

```
Please condense the following product title into a concise 2-4 word summary that captures the essential product identity. Only output the condensed title, nothing else.

Original title: [PRODUCT TITLE]

Condensed title:
```

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-1-7B |
| Max Input Length | 256 tokens |
| Max Output Length | 20 tokens |
| Temperature | 0.7 |
| Top-p | 0.9 |
| Batch Size | 64 |

## Data Sources

- **Input**: `meta_All_Beauty.jsonl` (Amazon Beauty metadata)
- **Output**: `item_titles_unique.json`

## Statistics

| Metric | Value |
|--------|-------|
| Total items | 112,590 |
| Items processed | 112,590 |
| Unique titles after regeneration | 28,344 |
| Initial duplicates fixed | ~4,000 |

## Notes

1. **Title Quality**: Some titles may contain placeholders like `[title1]`, `[title2]` due to LLM generation patterns. These can be regenerated if needed.

2. **GPU Memory**: The HuggingFace version requires ~16GB GPU memory. vLLM version is more memory efficient.

3. **Execution Time**:
   - Initial generation: ~2-3 hours
   - Duplicate regeneration (fast): ~30 minutes
   - Duplicate regeneration (slow): ~2 hours

4. **Downstream Use**: The condensed titles are used in `LLM_Rec_Data_Preparation/` for creating training data.

## Example Transformations

| Original Title | Condensed Title |
|----------------|-----------------|
| "Premium Organic Moisturizing Face Cream for Dry Skin with Vitamin E and Hyaluronic Acid" | "Moisturizing Face Cream" |
| "Professional Hair Straightener Flat Iron with Ceramic Plates and Adjustable Temperature" | "Ceramic Hair Straightener" |
| "Natural Beeswax Lip Balm Set - 6 Pack - Assorted Flavors" | "Beeswax Lip Balm Set" |
