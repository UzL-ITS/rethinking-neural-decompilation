# RQ1: Token Rank Analysis for LLM4Decompile

## Research Question

When LLM4Decompile does not predict the correct next source-code token as its
first choice, how highly is the correct token still ranked in the model's
output distribution?

The objective of RQ1 is to test whether the correct token is usually present
within the model's top-5 predictions. We evaluate this question across two
input representations, two datasets, and two LLM4Decompile models. We also use
tree-sitter to identify which C token types are easiest and hardest for the
models.

## Configurations

| ID | Input representation | Dataset | Model |
|---|---|---|---|
| C1 | Ghidra pseudo-code | HumanEval-Decompile | `LLM4Binary/llm4decompile-9b-v2` |
| C2 | Assembly | HumanEval-Decompile | `LLM4Binary/llm4decompile-6.7b-v1.5` |
| C3 | Ghidra/decompiler text | ExeBench | `LLM4Binary/llm4decompile-9b-v2` |
| C4 | Assembly | ExeBench | `LLM4Binary/llm4decompile-6.7b-v1.5` |

For each configuration, the experiment uses:

- 100 samples
- Random seed 42
- At most 128 reference target tokens per sample
- One teacher-forced model forward pass per sample
- Vectorized rank computation over the model vocabulary

## Method

For every sample, the decompiler representation is inserted into the
LLM4Decompile prompt and the original C function is used as the reference
target:

```text
# This is the assembly code:
<decompiler input>
# What is the source code?
<reference source function>
```

The prompt and reference target are tokenized separately and concatenated:

```text
[prompt tokens, target token 0, target token 1, ..., target token n-1]
```

With causal language modeling, the logits at position `i` predict the token at
position `i + 1`. If `p` is the prompt length, logits beginning at `p - 1`
therefore predict the reference target tokens. For each reference token `t_i`,
the experiment computes:

```text
rank(t_i) = 1 + count(vocabulary tokens with a higher score than t_i)
```

Interpretation:

- Rank 1: the correct token is the model's first choice.
- Rank <= 5: the correct token is available within top-5 decoding.
- Rank <= 10 or <= 100: the model assigns some probability to the correct
  token, but ranks it below stronger alternatives.
- Rank > 100: the correct token is poorly represented at that position.

All target positions are evaluated in one vectorized GPU operation after the
forward pass.

### Teacher-Forcing Scope

This experiment evaluates next-token distributions under teacher forcing. At
each position, the model receives the correct reference prefix, not its own
previous generated tokens. The results therefore measure local token-level
uncertainty and should not be interpreted as complete-program accuracy or
functional correctness.

## Token-Type Ablation

The ablation parses each reference C function with tree-sitter and maps each
decoded model token to a C syntax category. It reports the following for every
node type:

- Number of mapped tokens
- Median and mean rank
- Rank-1 percentage
- Top-5 and top-10 percentages
- Percentage with rank greater than 100 or 1,000

Examples of categories include `identifier`, `type_identifier`,
`keyword_qualifier`, `primitive_type`, `arithmetic_op`, `paren`, `brace`,
`comment`, and `string_literal`.

## Repository Layout

The publication artifact contains source code and notebooks only:

```text
RQ1_correct_token_idx/
|- README.md
|- token_rank_analysis.py
|- ablation/
|  `- token_rank_by_nodetype.py
`- solution/
   |- token_rank_analysis_rq1_c1_c2.ipynb
   |- exebench_token_analysis_rq1_c3_c4.ipynb
   |- token_ablation_c1.ipynb
   |- token_ablation_c2.ipynb
   |- token_ablation_c3.ipynb
   `- token_ablation_c4.ipynb
```

The two overall-analysis notebooks contain the configuration-specific token
rank experiments. The four ablation notebooks contain the corresponding
tree-sitter analyses. The root Python files provide standalone versions of the
core rank and node-type analysis logic.

Generated images, NPZ arrays, JSON summaries, and executed notebook outputs are
not included in the code-only artifact.

## Requirements

Recommended environment:

- Python 3.10+
- CUDA-capable GPU
- PyTorch
- Transformers
- Datasets
- NumPy
- Pandas
- Matplotlib
- tqdm
- Requests
- Jupyter
- tree-sitter and the tree-sitter C grammar

Example installation:

```bash
python -m pip install torch transformers datasets numpy pandas matplotlib \
  tqdm requests jupyter tree-sitter tree-sitter-c
```

The 9B model requires more GPU memory than the 6.7B model. The notebooks use
`float16` or `bfloat16` and should be run one configuration at a time.

## Dataset Preparation

### C1: HumanEval-Decompile Ghidra

The C1 notebook loads `LLM4Binary/decompile-eval` from Hugging Face and uses:

```text
input:  ghidra_pseudo
target: func
```

### C2: HumanEval-Decompile Assembly

The C2 overall experiment uses
`decompile-eval-executable-gcc-obj.json` and reads:

```text
input:  input_asm_prompt
target: c_func
```

Update the JSON URL or local path in
`solution/token_rank_analysis_rq1_c1_c2.ipynb` if necessary.

### C3 and C4: ExeBench

The ExeBench notebook currently loads a local Arrow shard. Update this path:

```python
Dataset.from_file("/path/to/exebench/data-00000-of-00004.arrow")
```

The configuration fields used by the notebooks are:

```text
C3 input:  c_code_unobf
C4 input:  assembly_unobf
target:    org_function
```

The local ExeBench data must provide these fields before the notebooks are run.

## Running the Experiment

From the repository root:

```bash
cd RQ1_correct_token_idx
export CUDA_VISIBLE_DEVICES=0
jupyter lab
```

Run the notebooks in this order:

```text
1. solution/token_rank_analysis_rq1_c1_c2.ipynb
   - Run the C1 cells with the 9B-v2 model.
   - Release GPU memory.
   - Run the C2 cells with the 6.7B-v1.5 model.

2. solution/exebench_token_analysis_rq1_c3_c4.ipynb
   - Run C3 with ExeBench decompiler text and the 9B-v2 model.
   - Release GPU memory.
   - Run C4 with ExeBench assembly and the 6.7B-v1.5 model.

3. solution/token_ablation_c1.ipynb
4. solution/token_ablation_c2.ipynb
5. solution/token_ablation_c3.ipynb
6. solution/token_ablation_c4.ipynb
```

Before running, replace the absolute dataset and output paths in the notebooks
with paths valid for the local environment.

### C4 Code Check

The current saved `exebench_token_analysis_rq1_c3_c4.ipynb` contains the C3
overall-analysis code but does not retain the complete C4 overall-analysis
cell. Before submitting the code artifact, restore the C4 cell using:

```text
model:  LLM4Binary/llm4decompile-6.7b-v1.5
input:  assembly_unobf
target: org_function
output: output_rq1_c4
```

Without that cell, the C4 result is documented but cannot be regenerated from
the submitted notebook.

## Generated Outputs

Each overall token-rank run generates:

```text
token_rank_analysis.png
token_rank_data.npz
```

The NPZ file contains:

```text
positions
median_ranks
mean_ranks
min_ranks
max_ranks
all_ranks
```

Each tree-sitter ablation generates:

```text
token_rank_by_nodetype.png
token_rank_by_nodetype.json
```

These files are runtime artifacts and are intentionally excluded from the
code-only publication repository.

## Unified Token-Rank Results

The central RQ1 result is that the correct token is within the top five for at
least 93.7% of evaluated positions in every configuration.

| Config | Evaluated tokens | Median rank | Rank = 1 | Rank <= 5 | Rank <= 10 | Rank <= 100 |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 11,441 | 1.0 | 80.9% | 93.7% | 95.6% | 98.7% |
| C2 | 11,208 | 1.0 | 87.5% | 97.2% | 97.9% | 99.0% |
| C3 | 9,071 | 1.0 | 87.3% | 97.1% | 98.1% | 99.4% |
| C4 | 9,027 | 1.0 | 82.3% | 94.6% | 96.2% | 98.7% |

The top-5 result directly answers RQ1:

```text
Minimum top-5 coverage across C1-C4: 93.7%
Maximum top-5 coverage across C1-C4: 97.2%
```

## Unified Token-Type Ablation Results

The following values come from the four tree-sitter ablation runs:

| Config | Mapped tokens | Mean rank | Median rank | Rank = 1 | Rank <= 10 |
|---|---:|---:|---:|---:|---:|
| C1 | 11,441 | 10.27 | 1.0 | 80.6% | 95.6% |
| C2 | 11,469 | 10.38 | 1.0 | 77.0% | 94.3% |
| C3 | 9,071 | 5.96 | 1.0 | 87.1% | 98.1% |
| C4 | 9,027 | 10.08 | 1.0 | 82.2% | 96.2% |

Token types with the highest mean ranks among categories with at least 50
observations were:

| Config | Harder token types |
|---|---|
| C1 | `keyword_qualifier`, `comment`, `type_identifier`, `identifier` |
| C2 | `comment`, `keyword_qualifier`, `type_identifier`, `identifier` |
| C3 | `keyword_qualifier`, `identifier`, `preproc_arg` |
| C4 | `keyword_qualifier`, `identifier`, `keyword_control` |

Structural categories such as parentheses, commas, braces, arithmetic
operators, and newlines generally have low mean ranks and high top-10 rates.
Identifiers and qualifiers are harder because their correct form depends more
strongly on program-specific semantics and naming choices.

## Key Findings

1. **The correct token is usually in the top five.** Top-5 coverage ranges from
   93.7% to 97.2% across all configurations.
2. **The correct token is most often ranked first.** Rank-1 accuracy ranges
   from 80.9% to 87.5%.
3. **Severe token-ranking failures are uncommon.** At least 98.7% of reference
   tokens are ranked within the top 100.
4. **The result generalizes across representations and datasets.** Both Ghidra
   and assembly inputs show high top-5 coverage on HumanEval-Decompile and
   ExeBench.
5. **Semantic token categories are harder than structural syntax.** Identifiers,
   qualifiers, and comments produce larger rank errors than common punctuation
   and operators.


