# Rethinking Neural Decompilation

Implementations and reproduction scripts for all research questions in the
paper. Each experiment has additional setup and execution instructions in its
own README.

## Experiments

| Folder | RQ | Summary | Key Result |
|--------|----|---------|------------|
| [RQ1_correct_token_idx](RQ1_correct_token_idx/) | Token rank analysis | Where does the correct token rank in the model's predictions? | 79.9% exact match, 95.3% top-10 |
| [RQ2](RQ2/) | Candidate reranking | Which selection strategy best improves pass@1 from pass@k? | Execution-guided: 71.1% mean pass@1; negative perplexity: 66.0% |
| [RQ3](RQ3/) | Obfuscation fine-tuning | Can fine-tuning improve decompilation of obfuscated code? | Fine-tuned + execution-guided: 88.8% pass@1 (+9.3pp) |

## RQ3 Dataset

The obfuscated ExeBench dataset used for RQ3 is publicly available on the
[Hugging Face Hub](https://huggingface.co/datasets/gautiiii/ExeBench-Obfuscated)
as `gautiiii/ExeBench-Obfuscated`. It provides separate `train` and `test`
splits.

Import both splits:

```python
from datasets import load_dataset

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

Create an environment and install the shared dependencies from the repository
root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run each experiment independently by following its complete pipeline:

- [RQ1 token-rank analysis](RQ1_correct_token_idx/README.md)
- [RQ2 candidate reranking](RQ2/README.md)
- [RQ2b advanced code reranking](RQ2b_Advanced_Code_Reranking/README.md)
- [RQ3 obfuscation fine-tuning and selection](RQ3/README.md)

Select the required GPU before starting a model-dependent stage, for example
with `export CUDA_VISIBLE_DEVICES=0` on Linux. Evaluation-only scripts may
depend on candidate and metric caches produced by preceding pipeline stages.

## Models

| Model | Used In | Purpose |
|-------|---------|---------|
| LLM4Binary/llm4decompile-6.7b-v2 | RQ1 | Token-rank analysis in the supplied RQ1 implementation |
| LLM4Binary/llm4decompile-9b-v2 | RQ2 C1/C3, RQ3 | Ghidra-based candidate generation, evaluation, and fine-tuning base |
| LLM4Binary/llm4decompile-6.7b-v1.5 | RQ2 C2/C4 | Assembly-based candidate generation and evaluation |
| microsoft/unixcoder-base | RQ2 | Code embedding for selection strategies |

## Dependencies

See [requirements.txt](requirements.txt) for the shared dependencies. Core
packages are `torch`, `transformers`, `vllm`, and `datasets`; experiment-specific
packages and system requirements are documented in the corresponding README.
