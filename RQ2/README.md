# RQ2: Candidate Reranking for Neural Decompilation

**Research question:** How effectively can generate-then-select strategies
improve pass@1 by choosing one decompilation from a pool of model-generated
candidates?

## Setup

- **Default script configuration:** C1 with
  `LLM4Binary/llm4decompile-9b-v2`
- **Default evaluation set:** 656 HumanEval-Decompile inputs derived from 164
  functions compiled at four optimization levels
- **Candidate pool:** 100 candidates per input
- **Sampling temperatures:** 0.0, 0.3, 0.5, 0.7, 0.9, and 1.0
- **Maximum sequence length:** 4096 tokens
- **Default script seed:** 42
- **Primary paper metric:** pass@1 functional correctness
- **Additional implementation outputs:** compilation rate, normalized
  Levenshtein similarity, and CodeBLEU

The paper evaluates four configurations that pair each input representation
with the corresponding LLM4Decompile checkpoint:

| ID | Input | Benchmark | Model | Inputs per seed |
|----|-------|-----------|-------|-----------------|
| C1 | Ghidra pseudo-code | HumanEval-Decompile | LLM4Decompile-Ref-9B-v2 | 656 |
| C2 | Assembly | HumanEval-Decompile | LLM4Decompile-End-6.7B-v1.5 | 656 |
| C3 | Ghidra pseudo-code | ExeBench | LLM4Decompile-Ref-9B-v2 | 1,000 |
| C4 | Assembly | ExeBench | LLM4Decompile-End-6.7B-v1.5 | 1,000 |

For HumanEval-Decompile, the 656 inputs correspond to 164 functions evaluated
at `O0`, `O1`, `O2`, and `O3`. For ExeBench, 1,000 inputs are sampled uniformly
for evaluation. The paper repeats candidate generation and evaluation with five
seeds and reports the per-configuration mean and standard deviation. The code in
this directory defaults to C1 and seed 42; reproduce the complete protocol by
running the pipeline separately for every configuration and generation seed.

The main configuration values are defined near the top of
`selection_pipeline.py`, `evaluate_all_metrics.py`, and
`evaluate_selection.py`. Keep `num_samples`, `num_candidates`, `seed`, and
`cache_dir` consistent across these scripts when changing configurations.

## Requirements

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Candidate generation and model-based scoring require a CUDA-capable GPU.
Compilation and functional testing additionally require Linux, GCC, and the
included `run-sandboxed.sh` wrapper. Generated programs are untrusted code and
should only be executed in an isolated environment.

The default C1 scripts expect the evaluation dataset at:

```text
../../LLM4Decompile/legacy-test/decompile-eval-executable-gcc-ghidra.json
```

Each input record must provide `input_asm_prompt`, `c_func`, and `c_test`.
Update `data_path` in `selection_pipeline.py` and
`train_contrastive_unixcoder.py` if the dataset is stored elsewhere. C2 and C4
require assembly inputs and the 6.7B End checkpoint, while C3 requires the
Ghidra-derived ExeBench evaluation set. Update the model, dataset, sample count,
and cache directory together before running a different configuration.

## Pipeline

```text
Step 0 (optional): train_contrastive_unixcoder.py -> checkpoints/contrastive_unixcoder_best.pt
Step 1:            selection_pipeline.py          -> cache/candidates_*_seed*.pkl
Step 2:            evaluate_all_metrics.py        -> cache/phase*_seed*.pkl
Step 3:            evaluate_selection.py          -> selection_evaluation_results_seed*.json
```

`selection_pipeline.py` generates the candidate pool and records generation
timing and a candidate fingerprint. `evaluate_all_metrics.py` incrementally
caches compilation labels, selector scores, perplexities, and functional test
outcomes. `evaluate_selection.py` reconstructs each candidate from these
caches, applies every selector, and computes the final metrics.

### Full run

Run all commands from this directory so that relative cache and sandbox paths
resolve correctly:

```bash
cd RQ2
export CUDA_VISIBLE_DEVICES=0

# Skip only when checkpoints/contrastive_unixcoder_best.pt already exists.
python train_contrastive_unixcoder.py

# Generate the candidate pool.
python selection_pipeline.py

# Compute compilation, selector, perplexity, and test-result caches.
python evaluate_all_metrics.py

# Evaluate selectors and write the final JSON summary.
python evaluate_selection.py
```

### Re-evaluate existing caches

When the candidate and phase caches already exist, the final evaluation can be
regenerated without candidate generation:

```bash
python evaluate_selection.py
```

The scripts reuse existing cache entries. Set `force_regenerate = True` in
`selection_pipeline.py` only when the candidate pool must be regenerated.

## Selection Strategies

| Category | Selector | Description |
|----------|----------|-------------|
| Baseline | `random` | Random candidate from the compiling subset. |
| Upper bound | `oracle` | Selects a passing candidate whenever one exists. |
| Likelihood | `neg_perplexity` | Selects the candidate with the lowest model perplexity. |
| Consensus | `mbr_consensus` | Selects the candidate most similar to the remaining pool. |
| Execution | `execution_guided` | Ranks candidates using partial functional-test feedback. |
| Text | `edit_similarity` | Character-level similarity to the decompiler input. |
| Text | `token_f1` | Token-overlap F1 score. |
| Text | `bigram_sim`, `trigram_sim` | N-gram Jaccard similarities. |
| Structure | `ast_node_sim` | AST node-distribution similarity. |
| Structure | `control_flow_sim` | Control-flow construct similarity. |
| Structure | `identifier_sim` | Identifier-set similarity. |
| Embedding | `unixcoder_base` | Base UniXcoder embedding similarity. |
| Embedding | `unixcoder_finetuned` | Contrastively fine-tuned UniXcoder similarity. |
| Representation | `repe_correctness` | Hidden-state projection onto a learned correctness direction. |
| Vote | `majority_vote_normalized_compile_only` | Most frequent whitespace-normalized compiling candidate. |
| Vote | `majority_vote_exact_compile_only` | Most frequent exact compiling candidate. |
| Vote | `majority_vote_normalized_all` | Most frequent normalized candidate in the full pool. |
| Vote | `majority_vote_exact_all` | Most frequent exact candidate in the full pool. |
| Ensemble | `ensemble_text` | Weighted text-metric combination. |
| Ensemble | `ensemble_unixcoder_ft` | Fine-tuned UniXcoder and text combination. |
| Ensemble | `ensemble_perplexity` | Perplexity and text combination. |

Except for the two majority-vote `all` variants, selectors operate on candidates
marked as compiling. The `all` variants intentionally vote over the complete
pool, including non-compiling candidates. Inputs for which no candidate
compiles are excluded from the compile-filtered selector evaluation and
reported separately.

## Optional Analyses

### Majority vote

Evaluate exact and normalized majority-vote variants and merge them into the
selection summary:

```bash
python majority_vote_evaluation.py \
  --in-pkl cache/candidates_656_100_seed42.pkl \
  --selection-results selection_evaluation_results_seed42.json
```

This produces `cache/majority_vote_results.json` and updates the specified
selection-results file. Existing candidate and phase caches are read but not
modified.

### RepE correctness

Compute hidden-state correctness scores from cached compilation and test
labels:

```bash
python compute_repe_correctness.py \
  --cache-dir cache \
  --seed 42 \
  --num-samples 656 \
  --num-candidates 100
```

This analysis requires the candidate, compilation, and test-result caches from
the core pipeline. It writes a RepE score cache, a fitted direction checkpoint,
and a standalone evaluation JSON.

### Oracle@k curve

Compute the oracle upper bound as the candidate-pool size increases without
rerunning generation or tests:

```bash
python oracle_curve_from_cache.py \
  --cache-dir cache \
  --candidates-file cache/candidates_656_100_seed42.pkl
```

The script writes `oracle_curve.json` and `oracle_curve.csv` in the cache
directory unless alternative output paths are supplied.

## Files

```text
RQ2/
|- README.md
|- selection_pipeline.py           # Candidate generation and initial scoring
|- evaluate_all_metrics.py         # Cached selector and functional evaluation
|- evaluate_selection.py           # Final pass@1, Levenshtein, and CodeBLEU
|- train_contrastive_unixcoder.py   # Optional contrastive UniXcoder training
|- unixcoder.py                     # UniXcoder model wrapper
|- majority_vote_evaluation.py      # Optional majority-vote selectors
|- compute_repe_correctness.py      # Optional RepE correctness selector
|- oracle_curve_from_cache.py       # Optional oracle@k analysis
|- run-sandboxed.sh                 # Isolated functional-test wrapper
|- checkpoints/                     # Created for trained selector checkpoints
`- cache/                           # Created for candidate and phase caches
```

## Cache Files

For the default configuration, the principal cache files use the seed suffix
`_seed42`:

| Cache | Purpose |
|-------|---------|
| `candidates_656_100_seed42.pkl` | Generated candidate pool. |
| `generation_timing_656_100_seed42.json` | Candidate-generation timing. |
| `candidate_fingerprint_656_100_seed42.json` | Candidate integrity metadata. |
| `phase1_compilation_seed42.pkl` | Compilation labels. |
| `phase2_text_metrics_seed42.pkl` | Text and structural scores. |
| `phase2_5_mbr_seed42.pkl` | MBR consensus scores. |
| `phase3_unixcoder_seed42.pkl` | UniXcoder similarities. |
| `phase3_5_execution_guided_seed42.pkl` | Execution-guided scores. |
| `phase3_perplexity_seed42.pkl` | Perplexity scores. |
| `phase4_test_results_seed42.pkl` | Functional test outcomes. |

Cache files are configuration-specific. Do not reuse them after changing the
dataset, candidate count, generation seed, or candidate ordering.
