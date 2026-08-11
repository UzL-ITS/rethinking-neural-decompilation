# RQ1: Token Rank Analysis for LLM4Decompile

**RQ**: When LLM4Decompile predicts the wrong token, how far off is it? Is the correct token ranked 2nd, 10th, or 1000th in the model's output distribution?

## Setup

- **Model**: LLM4Binary/llm4decompile-6.7b-v2
- **Dataset**: LLM4Binary/decompile-eval (100 random samples, seed=42)
- **Method**: Single forward pass per sample with teacher forcing (vectorized rank computation)
- **Max target tokens**: 128 per sample

## Pipeline

```
Step 1:  token_rank_analysis.py  -> token_rank_data.npz + token_rank_analysis.png
```

### Run:
```bash
cd v2/RQ1_correct_token_idx
export CUDA_VISIBLE_DEVICES=2,3
python token_rank_analysis.py
```

### Ablation (by AST node type):
```bash
python ablation/token_rank_by_nodetype.py
```

## Results

### Overall Token Rank Distribution (100 samples, 11,469 tokens)

| Metric | Value |
|--------|-------|
| Median rank | **1.0** |
| Mean rank | 13.3 |
| Rank = 1 (exact match) | **79.9%** |
| Rank <= 5 | 93.0% |
| Rank <= 10 | 95.3% |
| Rank <= 100 | 98.8% |
| Rank > 100 | 1.2% |
| Rank > 1000 | 0.1% |

### Ablation: Hardest Token Types (by mean rank)

| Node Type | Mean Rank | Median | Exact Match | Count |
|-----------|-----------|--------|-------------|-------|
| comment | 140.4 | 1.0 | 50.9% | 529 |
| unary_op | 61.6 | 1.0 | 81.5% | 27 |
| keyword_qualifier | 39.2 | 2.0 | 44.4% | 108 |
| type_identifier | 21.9 | 1.0 | 69.7% | 765 |
| identifier | 11.2 | 1.0 | 77.6% | 4163 |

### Ablation: Easiest Token Types (100% exact match)

escape_sequence, character, defined, #endif (all median=1, mean=1)

## Files

```
RQ1_correct_token_idx/
├── README.md
├── token_rank_analysis.py       # Main experiment: rank of correct token per position
├── token_rank_analysis.png      # 4-panel plot (median by position, histogram, CDF)
├── token_rank_data.npz          # Raw numpy data (ranks by position)
└── ablation/
    ├── token_rank_by_nodetype.py    # Breakdown by tree-sitter AST node type
    ├── token_rank_by_nodetype.png   # Node-type visualization
    └── token_rank_by_nodetype.json  # Per-node-type statistics (41 types)
```

## Key Findings

1. **The model is mostly right**: 79.9% of tokens are predicted exactly (rank 1)
2. **When wrong, usually close**: 93.0% of tokens are in the top 5
3. **Only 1.2% are truly wrong** (rank > 100), suggesting the model understands code structure
4. **Comments are hardest** (mean rank 140.4) - model struggles with free-form text in code
5. **Identifiers are the biggest challenge at scale**: though 77.6% exact, they represent 36% of all tokens
6. **Structural tokens are easy**: brackets, semicolons, keywords have near-perfect prediction

## Method Details

For each sample, we concatenate `[prompt_tokens, target_tokens]` and run a single forward pass.
The logits at position `prompt_len-1` through `prompt_len-1+target_len` predict the target tokens (teacher forcing).
For each position, we compute `rank = 1 + count(tokens with higher probability than correct token)`.
This is fully vectorized on GPU, making it ~100x faster than per-token forward passes.
