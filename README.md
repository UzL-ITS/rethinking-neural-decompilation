# v2: Paper Experiments

Clean, self-contained implementations of all research questions for the paper.

## Experiments

| Folder | RQ | Summary | Key Result |
|--------|----|---------|------------|
| [RQ1_correct_token_idx](RQ1_correct_token_idx/) | Token rank analysis | Where does the correct token rank in the model's predictions? | 79.9% exact match, 95.3% top-10 |
| [RQ2](RQ2/) | Candidate reranking | Which selection strategy best improves pass@1 from pass@k? | Perplexity selection: 72.5% pass@1 (+7pp) |
| [RQ3](RQ3/) | Obfuscation fine-tuning | Can fine-tuning improve decompilation of obfuscated code? | +27.8pp pass@1 (17.7% -> 45.6%) |

## RQ3 Dataset

The obfuscated ExeBench dataset used for RQ3 is publicly available on the
[Hugging Face Hub](https://huggingface.co/datasets/gautiiii/ExeBench-Obfuscated)
as `gautiiii/ExeBench-Obfuscated`. It provides separate `train` and `test`
splits.

Import both splits:

```python
train_dataset = load_dataset(
    "gautiiii/ExeBench-Obfuscated",
    split="train",
)
test_dataset = load_dataset(
    "gautiiii/ExeBench-Obfuscated",
    split="test",
)
```

Each sample contains the following fields:

| Field | Description |
|-------|-------------|
| `task_id` | Stable identifier for the sample. |
| `func_name` | Original function name. |
| `c_func` | Reference C function used as the target output. |
| `input_asm_prompt` | Decompiled obfuscated function supplied to the model. |
| `type` | Compiler optimization level, such as `-O0` or `-O3`. |
| `obf_pass` | Applied OLLVM obfuscation pass (`bcf`, `fla`, `fla+split`, or `sub`). |
| `path` | Source-repository path associated with the function. |
| `c_test` | C test harness used for functional evaluation. |
| `real_io_pairs` | Input/output examples associated with the function. |
| `real_iospec` | Serialized input/output specification used by the evaluation pipeline. |

## Quick Start

```bash
# Setup
source ../venv/bin/activate
export CUDA_VISIBLE_DEVICES=2,3
pip install -r requirements.txt

# Each experiment can be run independently from its folder
cd RQ1_correct_token_idx && python token_rank_analysis.py
cd RQ2 && python evaluate_selection.py
cd RQ3 && python plot_obfuscated_results.py
```

## Models

| Model | Used In | Purpose |
|-------|---------|---------|
| LLM4Binary/llm4decompile-6.7b-v2 | RQ1 | Token rank analysis |
| LLM4Binary/llm4decompile-9b-v2 | RQ2, RQ3 | Candidate generation, evaluation, fine-tuning base |
| microsoft/unixcoder-base | RQ2 | Code embedding for selection strategies |

## Dependencies

See [requirements.txt](requirements.txt) for the full list. Core: `torch`, `transformers`, `vllm`, `datasets`.
