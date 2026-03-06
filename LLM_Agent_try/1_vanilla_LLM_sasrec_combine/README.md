# LLM + SASRec Fusion Experiments

This directory contains experiments combining LLM-based recommendation with SASRec scores for improved performance.

## Motivation

- **SASRec**: Good at capturing sequential patterns (HR@1=9.22%, HR@10=13.03%)
- **LLM (condensed titles)**: Good at capturing semantic information (HR@1=9.26%, HR@10=12.18%)

By combining both approaches, we aim to leverage their complementary strengths.

## Two Fusion Approaches

### Approach 1: Prompt Augmentation
- Include SASRec's top-k predictions as additional context in LLM prompt
- LLM can use this information to refine its prediction

### Approach 2: Score Fusion
- Combine SASRec scores with LLM embedding similarity
- Final score = α * SASRec_score + β * LLM_similarity
- Tunable weights to balance both signals

## Key Challenges

1. **Sequence Alignment**: LLM uses original item_ids, SASRec uses internal IDs
   - Solution: Use item_mapping.json to create bidirectional mappings

2. **Score Calibration**: SASRec scores and LLM similarities have different scales
   - Solution: Normalize both to [0, 1] range before fusion

## Files

```
1_vanilla_LLM_sasrec_combine/
├── README.md                    # This file
├── approach1_prompt_augment.py  # Prompt augmentation experiment
├── approach2_score_fusion.py    # Score fusion experiment
└── results/                     # Experiment results
```

## Usage

```bash
# Approach 1: Prompt Augmentation
CUDA_VISIBLE_DEVICES=6,7 python approach1_prompt_augment.py

# Approach 2: Score Fusion
CUDA_VISIBLE_DEVICES=6,7 python approach2_score_fusion.py
```

## Expected Improvements

Target: HR@10 > 13.03% (better than both individual models)
