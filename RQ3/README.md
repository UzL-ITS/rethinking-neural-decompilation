# RQ3: Decompilation of Obfuscated Binaries

**Research question:** Does multi-candidate selection, with targeted
fine-tuning, enable neural decompilation of obfuscated binaries?

## Setup

- **Base model:** `LLM4Binary/llm4decompile-9b-v2`
- **Input representation:** Ghidra pseudo-code recovered from obfuscated
  binaries
- **Training:** one epoch on the ExeBench-Obfuscated training split
- **Evaluation subset:** 1,000 test samples selected with `random_state=42`
- **Candidate pool:** 100 candidates per input for selection evaluation
- **Selection methods reported in the paper:** perplexity and execution-guided
- **Maximum sequence length:** 4096 tokens for prompt and generated output
- **Metrics:** pass@1, compilation rate, normalized Levenshtein similarity, and
  CodeBLEU

The same fixed test subset is used for the original and fine-tuned models. It
contains all four OLLVM transformations and all four compiler optimization
levels:

| Dimension | Category | Samples |
|-----------|----------|---------|
| Obfuscation pass | `bcf` | 233 |
| Obfuscation pass | `fla` | 208 |
| Obfuscation pass | `fla+split` | 251 |
| Obfuscation pass | `sub` | 308 |
| Optimization level | `O0` | 219 |
| Optimization level | `O1` | 262 |
| Optimization level | `O2` | 235 |
| Optimization level | `O3` | 284 |

Pass@1 is functional correctness: a selected program must compile and pass all
tests within the evaluation timeout. Samples for which no candidate compiles
remain in the denominator and therefore count as failures. Levenshtein and
CodeBLEU are averaged over the same 1,000 inputs, with a zero contribution when
no candidate is selected.

## Dataset

The dataset is publicly available from the Hugging Face Hub:

```python
from datasets import load_dataset

dataset = load_dataset("gautiiii/ExeBench-Obfuscated")
train_dataset = dataset["train"]
test_dataset = dataset["test"]
```

- **Repository:** [gautiiii/ExeBench-Obfuscated](https://huggingface.co/datasets/gautiiii/ExeBench-Obfuscated)
- **Training split:** 287,790 samples
- **Test split:** 74,857 samples
- **Total:** 362,647 samples

The corpus is split by source project rather than by individual function to
avoid cross-project leakage. Empty and trivial functions are removed, and each
sample is mapped back to its ExeBench source metadata and execution fields.
Important columns include `task_id`, `func_name`, `c_func`,
`input_asm_prompt`, `type`, `obf_pass`, `c_test`, `real_io_pairs`,
`real_iospec`, and `path`.

The current scripts read materialized JSON files. Set `TRAIN_DATA` and
`TEST_DATA` in `train_obfuscation.py`, and `obfuscated_data` in each evaluation
script, to the corresponding local train and test files before running the
pipeline. The JSON test records must retain `task_id` so that every stage can
reuse the exact sample manifest selected by the oracle analysis.

## Requirements

Install the shared dependencies from the repository root:

```bash
pip install -r requirements.txt
pip install pandas bitsandbytes accelerate
```

Fine-tuning, vLLM candidate generation, and perplexity scoring require a
CUDA-capable GPU. Compilation and functional testing require Linux, GCC, and
the included `run-sandboxed.sh` wrapper. The sandbox uses Bubblewrap when it is
available. Generated programs are untrusted code and must not be executed
outside an isolated environment.

## Data Preparation

`train_obfuscation.py` applies the following checks before training:

1. Remove duplicate samples using normalized Ghidra pseudo-code hashes.
2. Remove training functions whose normalized reference source overlaps with
   the test split.
3. Remove training and test records whose complete prompt-plus-target sequence
   exceeds 4096 tokens.
4. Mask prompt tokens in the training labels so that loss is computed only on
   the target source code.

The cleaned samples are used to fine-tune the base checkpoint for one epoch
with bfloat16 precision, gradient checkpointing, batch size 1, gradient
accumulation over four steps, and a learning rate of `2e-5`.

## Pipeline

```text
Step 0: train_obfuscation.py
        -> checkpoints/llm4decompile-9b-obf-epoch1/

Step 1: analyze_obfuscated_oracle.py
        -> cache_oracle/oracle_results.json
        -> fixed 1,000-sample manifest and pass@256 diagnostic

Step 2 (optional): plot_obfuscated_results.py
        -> plots/obfuscated_heatmap.png
        -> plots/obfuscated_bars.png
        -> plots/obfuscated_detailed.png

Step 3 (optional): evaluate_obfuscated.py
        -> results_greedy/summary.json
        -> greedy original-versus-fine-tuned baseline

Step 4: evaluate_obfuscated_selection_best.py
        -> results_selection/summary.json
        -> results_selection/latex_data.json
```

The pass@256 oracle stage is a diagnostic and sample-manifest stage; it is not
the 100-candidate selection result reported in the paper. The main paper
evaluation is produced by `evaluate_obfuscated_selection_best.py`, which uses
the 100-candidate pool for both the original and fine-tuned models.

### Full run

Run the commands from the `RQ3` directory so that relative sandbox, cache, and
checkpoint paths resolve correctly:

```bash
cd RQ3
export CUDA_VISIBLE_DEVICES=0

# Fine-tune the 9B base checkpoint for one epoch.
python train_obfuscation.py

# Select and persist the fixed 1,000-sample evaluation manifest.
python analyze_obfuscated_oracle.py

# Optional oracle plots.
python plot_obfuscated_results.py

# Optional single-candidate greedy baseline.
python evaluate_obfuscated.py

# Main 100-candidate selection evaluation.
python evaluate_obfuscated_selection_best.py
```

If the fine-tuned checkpoint already exists, Step 0 can be skipped. If
`cache_oracle/oracle_results.json` already contains the intended sample
manifest, Step 1 can also be skipped. Do not replace the manifest between
model evaluations, because that would compare the models on different inputs.

## Selection Methods

| Selector | Candidate pool | Description | Paper usage |
|----------|----------------|-------------|-------------|
| `perplexity` | Compiling candidates | Chooses the candidate with the lowest model perplexity conditioned on the Ghidra input. | Main static selector |
| `execution_guided` | Compiling candidates | Scores partial functional behavior using execution-derived checks and uses perplexity for tie-breaking. | Main dynamic selector |
| `mbr_consensus` | Compiling candidates | Chooses the candidate with the highest lexical and structural agreement with the candidate pool. | Supplementary script output |

`evaluate_obfuscated_selection.py` is the earlier perplexity-only evaluation.
`evaluate_obfuscated_selection_best.py` is the comprehensive version and
should be used to regenerate the paper-aligned comparison. Its default
candidate temperatures are 0.3, 0.5, 0.7, 0.9, and 1.0.

## Outputs and Caches

The comprehensive evaluation writes intermediate caches separately for the
original and fine-tuned models:

| Cache pattern | Purpose |
|---------------|---------|
| `candidates_<model>.pkl` | Generated 100-candidate pools. |
| `compilation_<model>.pkl` | Per-candidate compilation labels. |
| `perplexity_<model>.pkl` | Perplexity scores for compiling candidates. |
| `mbr_consensus_<model>.pkl` | MBR consensus scores. |
| `execution_guided_<model>.pkl` | Execution-guided scores. |
| `tests_<model>_<selector>.pkl` | Functional outcomes for selected candidates. |

`results_selection/summary.json` contains the complete metrics and selected
outputs for each model and selector. `results_selection/latex_data.json`
contains the compact overall, obfuscation-pass, optimization-level, and
selection-method breakdowns used for table generation.

Cache files are tied to the model checkpoint, sample manifest, candidate
ordering, and generation configuration. Remove or relocate incompatible caches
before changing any of these inputs.

## Files

```text
RQ3/
|- README.md
|- train_obfuscation.py                  # One-epoch obfuscation fine-tuning
|- analyze_obfuscated_oracle.py          # Pass@256 diagnostic and sample manifest
|- plot_obfuscated_results.py            # Oracle plots
|- evaluate_obfuscated.py                # Greedy baseline evaluation
|- evaluate_obfuscated_selection.py      # Perplexity-only selection evaluation
|- evaluate_obfuscated_selection_best.py # Comprehensive selection evaluation
|- eval_utils.py                         # Compilation, execution, and metric helpers
|- run-sandboxed.sh                      # Isolated functional-test wrapper
|- clib/                                 # C execution support library
|- nlohmann/                             # JSON header used by execution support
|- checkpoints/                          # Fine-tuned checkpoint output
|- cache_oracle/                         # Oracle and sample-manifest caches
|- cache_selection/                      # Candidate and selector caches
|- results_greedy/                       # Greedy evaluation outputs
|- results_selection/                    # Selection evaluation outputs
`- plots/                                # Oracle-analysis plots
```
