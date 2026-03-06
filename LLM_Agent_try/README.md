# LLM as Rec Agent: Amazon Beauty Exploratory Branch

## 1. This Folder Is Trying to Do

This directory is a focused side branch inside the larger `Fuxi-OneRec-new` project. The parent project already has a full multi-stage recommendation pipeline built around retrieval, SFT, and RLHF. By contrast, this folder asks a narrower question:

> Can an LLM act more directly as a recommendation agent, and can it be improved by combining its semantic understanding with SASRec's sequential signal?

The experiments here are built around the Amazon Beauty dataset and use a much smaller, faster loop than the full OneRec pipeline. The branch does not try to replace the main project. It is meant to validate concrete hypotheses before moving ideas back into the main system.

## 2. Core Research Questions

This branch explores five related questions:

1. How strong is a plain SASRec baseline on Beauty when trained locally with vanilla item IDs?
2. Can a general LLM directly recommend the next item from user history?
3. Does rewriting long product titles into short condensed titles make LLM-based recommendation substantially better?
4. If SASRec and LLM have complementary strengths, what is the simplest effective fusion method?
5. Can fusion be made more adaptive, for example by deciding when SASRec should dominate?

## 3. High-Level Findings

The main conclusions so far are already fairly clear:

- Plain LLM recommendation with original product titles performs poorly.
- Condensed titles help a lot. Short semantic labels are much better prompts than noisy raw metadata.
- SASRec and the LLM are complementary:
  - SASRec is stronger at sequence pattern learning.
  - The LLM is useful for semantic matching.
- The best working method so far is **score fusion**, not prompt augmentation.
- More adaptive fusion and learned gating are promising directions, but they are still exploratory and not yet the most stable part of this branch.

## 4. Best Results So Far

Dataset and evaluation setup used across the main comparison:

- Dataset: Amazon Beauty
- Number of aligned test samples for fusion experiments: 4,385
- Number of filtered items: 28,344
- Metrics: `HR@1`, `HR@5`, `HR@10`, `HR@20`, `NDCG`

### Main comparison

| Method | Best setting | HR@1 | HR@10 | Notes |
|---|---:|---:|---:|---|
| SASRec baseline | - | 9.22% | 13.03% | Strong sequential baseline |
| LLM with original titles | - | 5.13% | 7.65% | Clearly weak |
| LLM with condensed titles | - | 9.26% | 12.18% | Large jump over raw titles |
| Score Fusion | `alpha=0.7` | **10.95%** | **13.25%** | Current best |
| Prompt Augmentation | `k=20` | 5.56% | 7.73% | Did not work |

### Interpretation

- Condensed titles improved the LLM dramatically relative to original titles.
- Score fusion outperformed both standalone SASRec and standalone LLM.
- Prompt augmentation underperformed badly, suggesting that simply injecting SASRec candidates into the prompt can distract the model instead of helping it.

## 5. Experiment Flow

The branch is best read as a sequence of increasingly targeted experiments.

### Stage A: Build a local SASRec baseline

Directory: `vanilla_sasrec/`

This stage creates a clean sequential baseline on Amazon Beauty:

- detect sequence statistics
- preprocess interactions into train/val/test
- build a simple item-ID tokenizer
- train SASRec
- evaluate with generation-style top-k metrics

This gives the project a strong non-LLM anchor and provides:

- `item_mapping.json` for mapping between internal numeric IDs and original item IDs
- local checkpoints for downstream fusion experiments
- a clean reference point for whether LLM-based approaches are actually worth the added complexity

### Stage B: Convert product titles into condensed semantic labels

Directory: `use_Qwen3-1-7B_to_generate_title/`

Raw Beauty titles are long, noisy, repetitive, and often unsuitable for prompt-based recommendation. This stage uses Qwen3-1-7B to turn each product title into a concise 2-4 word label.

Example:

- original: `Premium Organic Moisturizing Face Cream for Dry Skin with Vitamin E...`
- condensed: `Moisturizing Face Cream`

This step is foundational for the whole branch because it makes the LLM-facing recommendation prompts much cleaner.

### Stage C: Prepare LLM recommendation data

Directory: `LLM_Rec_Data_Preparation/`

This stage converts user interaction sequences into natural-language prompts. Each sample contains:

- user history
- condensed item titles
- rating information
- truncated review text
- the expected next product title as target

The goal here is to represent recommendation as a text generation problem while keeping the history understandable for a general LLM.

### Stage D: Establish the LLM baseline

Directory: `LLM_baseline_Prediction/`

This stage asks the simplest LLM question:

> Given the user's purchase history, can the model directly generate the next product?

Two important lessons came out of this baseline:

- using original titles is a poor choice
- embedding-based matching is much faster than raw string matching

This stage supplies the semantic half of the later fusion experiments.

### Stage E: Combine SASRec and LLM

Directory: `1_vanilla_LLM_sasrec_combine/`

Two combination strategies are tried here.

#### 1. Prompt Augmentation

The idea is to let SASRec produce top-k candidates and append them to the LLM prompt.

Result:

- intuitive, but ineffective
- performance dropped far below the baseline

Likely reason:

- the extra candidate list adds noise and changes the LLM's generation behavior in an unhelpful way

#### 2. Score Fusion

This is the strongest result in the branch.

Basic recipe:

1. use SASRec to score all candidate items
2. let the LLM generate a prediction
3. embed the generated text
4. compare it against all item-title embeddings
5. fuse the two signals

Fusion formula:

```text
final_score = alpha * sasrec_score + (1 - alpha) * llm_similarity
```

Best setting:

- `alpha = 0.7`

This means the best result is obtained when the system trusts SASRec more, while still using the LLM as a semantic correction signal.

### Stage F: Explore adaptive fusion

Directory: `2_find_time_to_add_sasrec_score_into_LLM/`

This stage asks a subtler question:

> If score fusion works, when should SASRec matter more and when should the LLM matter more?

Two exploratory ideas appear here:

- perplexity-based adaptive fusion
- score-difference-based adaptive fusion

The stored result file currently shows that perplexity-based adaptive fusion reaches roughly the same range as strong fixed fusion, but does not clearly beat the best simple `alpha=0.7` setting. This makes it interesting, but not yet a decisive improvement.

### Stage G: Learn a gate network

Directory: `3_gate_to_add_sasrec_score_into_LLM/`

This stage moves from hand-designed heuristics to a learned policy:

- freeze the LLM and SASRec
- extract hidden states from Qwen
- train a small MLP gate
- predict how much SASRec should influence the final decision

There are two versions:

- `train_gate.py`: earlier direct training version
- `train_gate_v2.py`: more efficient version with feature precomputation and differentiable objectives

This is a natural next step after fixed score fusion, but it should still be treated as work in progress.

### Stage H: Future SFT direction

Directory: `4_finetune_to_upgrade_LLM/`

This directory exists as a placeholder for the next idea: instead of only post-hoc fusion, improve the LLM itself through task-specific fine-tuning. At the time of writing, this stage is not yet populated and should be treated as planned work rather than a finished experiment.

## 6. Directory Map

```text
temp_LLM_Agent_try/
├── README.md
├── AGENTS.md
├── vanilla_sasrec/                        # Local SASRec pipeline and checkpoints
├── use_Qwen3-1-7B_to_generate_title/      # Title condensation
├── LLM_Rec_Data_Preparation/              # Prompt-style rec data
├── LLM_baseline_Prediction/               # Direct LLM baseline
├── 1_vanilla_LLM_sasrec_combine/          # Prompt augment + score fusion
├── 2_find_time_to_add_sasrec_score_into_LLM/  # Adaptive fusion heuristics
├── 3_gate_to_add_sasrec_score_into_LLM/   # Learned gating
├── 4_finetune_to_upgrade_LLM/             # Planned next stage
└── 0_results_all_in_one/                  # Summaries and result snapshots
```

## 7. Dependencies on the Parent Project

This folder is self-contained only at the experiment level. It still depends heavily on the parent project:

- `../../Rec-Transformer`
  - for `sasrec`
- `../../utils`
  - for shared data collators and evaluation helpers
- `../../LLM4RecPart/models/Qwen3-1-7B`
  - for the base LLM checkpoint
- `../../Data/Amazons/data/All_Beauty.jsonl`
  - for raw Beauty interactions and metadata

So if a script fails here, the cause is often not in this folder alone. Path assumptions and parent-level checkpoints matter.

## 8. How to Reproduce the Main Path

If someone wants to understand this branch from the ground up, the most sensible order is:

1. `vanilla_sasrec/`
   - train and evaluate the local SASRec baseline
2. `use_Qwen3-1-7B_to_generate_title/`
   - generate condensed titles
3. `LLM_Rec_Data_Preparation/`
   - create prompt-based recommendation data
4. `LLM_baseline_Prediction/`
   - measure direct LLM recommendation quality
5. `1_vanilla_LLM_sasrec_combine/`
   - run prompt augmentation and score fusion
6. `2_find_time_to_add_sasrec_score_into_LLM/`
   - test adaptive fusion ideas
7. `3_gate_to_add_sasrec_score_into_LLM/`
   - explore learned gating

Representative commands:

```bash
cd vanilla_sasrec
bash run.sh

cd ../LLM_Rec_Data_Preparation
python prepare_llm_rec_data.py
python verify_data.py

cd ../LLM_baseline_Prediction
python prepare_and_evaluate_baseline_optimized.py

cd ../1_vanilla_LLM_sasrec_combine
CUDA_VISIBLE_DEVICES=6,7 python approach2_score_fusion.py
```

## 9. What This Branch Has Shown

The strongest claim supported by the existing results is:

> For Beauty recommendation, the LLM is most useful here as a semantic signal, not as a standalone recommender and not as a prompt-conditioned selector over SASRec candidates.

More concretely:

- sequence modeling alone is strong
- semantics alone can be decent if the textual representation is cleaned up
- simple score-level fusion is more reliable than prompt-level intervention
- the fixed strong baseline is already hard to beat

That gives this branch a clear value inside the larger OneRec project: it provides evidence about how LLMs should be inserted into recommendation, and just as importantly, how they should not be inserted.

## 10. Suggested Next Steps

The most reasonable next steps from the current state are:

1. make the gate experiments reproducible and log final metrics in `3_gate_to_add_sasrec_score_into_LLM/results/`
2. test whether light SFT in `4_finetune_to_upgrade_LLM/` can improve the LLM signal before fusion
3. move the strongest validated idea back into the main OneRec pipeline if it continues to hold on other datasets

## 11. Related Sub-READMEs

For implementation details, see the per-module notes:

- `vanilla_sasrec/README.md`
- `use_Qwen3-1-7B_to_generate_title/README.md`
- `LLM_Rec_Data_Preparation/README.md`
- `LLM_baseline_Prediction/README.md`
- `1_vanilla_LLM_sasrec_combine/README.md`
- `0_results_all_in_one/README.md`
