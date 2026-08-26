#!/usr/bin/env python3
"""
Comprehensive evaluation of LLM4Decompile on obfuscated samples with selection.

This script:
1. Generates 100 candidates per sample using vLLM with temperature sampling
2. Filters by compilation
3. Uses perplexity-based selection (lowest perplexity = best)
4. Evaluates both original and fine-tuned models
5. Breaks down results by obfuscation type and optimization level
6. Saves detailed JSON results for LaTeX export

Metrics:
- Pass@1 (functional correctness)
- Compile rate
- Levenshtein similarity
- CodeBLEU score
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import subprocess
import tempfile
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import pickle
import time
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

import Levenshtein
from codebleu import calc_codebleu
import shutil
import shlex
from concurrent.futures import ThreadPoolExecutor
import sys
import re
PROJECT_DIR = Path(__file__).resolve().parent
_SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # Data
    obfuscated_data: str = str(_SCRIPT_DIR / "../RQ3_obfuscation_finetuning/train-test-split_v2/test_non_empty_functions.json")
    selected_samples_file: str = str(_SCRIPT_DIR / "cache_oracle/oracle_results.json")

    # Models
    original_model: str = "LLM4Binary/llm4decompile-9b-v2"
    finetuned_model: str = str(_SCRIPT_DIR / "checkpoints/llm4decompile-9b-obf-epoch1")

    # Generation settings
    num_candidates: int = 100
    temperatures: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9, 1.0])
    max_new_tokens: int = 4096
    max_sequence_len: int = 4096

    # Evaluation
    num_workers: int = 64

    # Output
    output_dir: str = str(_SCRIPT_DIR / "results_selection")
    cache_dir: str = str(_SCRIPT_DIR / "cache_selection")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Candidate:
    """A generated decompilation candidate."""
    code: str
    temperature: float
    compiles: bool = False
    passes_tests: bool = False
    perplexity: float = float('inf')
    levenshtein: float = 0.0
    codebleu: float = 0.0


@dataclass
class Sample:
    """A test sample with multiple candidates."""
    idx: int
    obf_pass: str
    opt_level: str
    ghidra_code: str
    c_func: str
    c_test: str
    real_io_pairs: dict
    real_iospec: str
    candidates: List[Candidate] = field(default_factory=list)

    selected_compiles: bool = False
    selected_passes_tests: bool = False
    selected_levenshtein: float = 0.0
    selected_codebleu: float = 0.0


# =============================================================================
# Data Loading
# =============================================================================

def compute_total_sequence_length(ghidra_code: str, source_code: str, tokenizer) -> int:
    """Compute full prompt+target sequence length before truncation."""
    prompt = f"# This is the assembly code:\n{ghidra_code}\n# What is the source code?\n"
    target = f"{source_code}\n{tokenizer.eos_token}"

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
        return_tensors=None,
    )["input_ids"]

    target_ids = tokenizer(
        target,
        add_special_tokens=False,
        truncation=False,
        return_tensors=None,
    )["input_ids"]

    return len(prompt_ids) + len(target_ids)


def compute_prompt_length(ghidra_code: str, tokenizer) -> int:
    prompt = build_prompt(ghidra_code)
    return len(
        tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
            return_tensors=None,
        )["input_ids"]
    )

def load_obfuscated_data(config: Config) -> List[Sample]:
    """Load obfuscated test data."""
    print(f"Loading obfuscated data from {config.obfuscated_data}...")

    with open(config.obfuscated_data, 'r') as f:
        raw_data = json.load(f)

    if isinstance(raw_data, list):
        length_tokenizer = AutoTokenizer.from_pretrained(
            config.original_model,
            model_max_length=config.max_sequence_len,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
        if length_tokenizer.pad_token is None:
            length_tokenizer.pad_token = length_tokenizer.eos_token
            length_tokenizer.pad_token_id = length_tokenizer.eos_token_id

        df = pd.DataFrame(raw_data)
        print("Filtering samples by full sequence length...")
        valid_indices = []
        skipped_long = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering lengths"):
            total_len = compute_total_sequence_length(
                row['input_asm_prompt'],
                row['c_func'],
                length_tokenizer,
            )
            if total_len <= config.max_sequence_len:
                valid_indices.append(idx)
            else:
                skipped_long += 1
        df = df.iloc[valid_indices].copy()
        print(f"  Skipped {skipped_long:,} overlength samples")
        print(f"  Samples within length limit: {len(df):,}")

        selected_path = Path(config.selected_samples_file)
        if selected_path.exists():
            print(f"Loading selected sample manifest from {selected_path}")
            with open(selected_path, 'r', encoding='utf-8') as f:
                selected_data = json.load(f)
            selected_keys = {
                (
                    str(row['task_id']),
                    str(row['opt_level']),
                    str(row['obf_pass']),
                )
                for row in selected_data['detailed_results']
            }
            df_keys = {
                (
                    str(row['task_id']),
                    str(row.get('type', row.get('opt_level'))),
                    str(row['obf_pass']),
                )
                for _, row in df.iterrows()
            }
            missing_keys = selected_keys - df_keys
            matched_df = df[
                df.apply(
                    lambda row: (
                        str(row['task_id']),
                        str(row.get('type', row.get('opt_level'))),
                        str(row['obf_pass']),
                    ) in selected_keys,
                    axis=1,
                )
            ].copy()
            matched_keys = {
                (
                    str(row['task_id']),
                    str(row.get('type', row.get('opt_level'))),
                    str(row['obf_pass']),
                )
                for _, row in matched_df.iterrows()
            }
            extra_keys = matched_keys - selected_keys
            print(f"  Oracle manifest keys: {len(selected_keys):,}")
            print(f"  Matched dataset rows: {len(matched_df):,}")
            print(f"  Missing keys: {len(missing_keys):,}")
            print(f"  Extra keys: {len(extra_keys):,}")
            if missing_keys or extra_keys or len(matched_df) != len(selected_keys):
                raise ValueError("Selected samples do not exactly match oracle manifest")
            print("  Exact oracle sample match confirmed")
            df = matched_df
        else:
            df = df.sample(n=min(1000, len(df)), random_state=42).reset_index(drop=True)
            print("  Selected sample manifest not found; fell back to seeded random sampling")

        df = df.reset_index(drop=True)
        raw_data = df.to_dict(orient="records")

        # Save sampled task IDs to list to check the samples selected later.
        sampled_ids = df["task_id"].tolist()
        with open("sampled_ids_1000.txt", "w") as f:
            for x in sampled_ids:
                f.write(f"{x}\n")
            
    samples = []
    if isinstance(raw_data, list):
        num_samples = len(raw_data)
        print(f"  Found {num_samples} samples")

        for idx, row in enumerate(raw_data):
            sample = Sample(
                idx=idx,
                obf_pass=str(row['obf_pass']),
                opt_level=row.get('type', row.get('opt_level')),
                ghidra_code=row.get('input_asm_prompt', row.get('c_code', row.get('c_code_obf'))),
                c_func=row['c_func'],
                c_test=row['c_test'],
                real_io_pairs=row['real_io_pairs'],
                real_iospec=row['real_iospec']
            )
            samples.append(sample)
    elif isinstance(raw_data, dict):
        num_samples = len(raw_data['func_name'])
        print(f"  Found {num_samples} samples")

        for idx in range(num_samples):
            str_idx = str(idx)
            code_column = raw_data.get('input_asm_prompt', raw_data.get('c_code', raw_data.get('c_code_obf')))
            sample = Sample(
                idx=idx,
                obf_pass=str(raw_data['obf_pass'][str_idx]),
                opt_level=raw_data['type'][str_idx],
                ghidra_code=code_column[str_idx],
                c_func=raw_data['c_func'][str_idx],
                c_test=raw_data['c_test'][str_idx],
            )
            samples.append(sample)
    else:
        raise TypeError(f"Unsupported JSON structure: {type(raw_data).__name__}")

    print("\n  Obfuscation pass distribution:")
    obf_counts = {}
    for s in samples:
        obf_counts[s.obf_pass] = obf_counts.get(s.obf_pass, 0) + 1
    for k, v in sorted(obf_counts.items()):
        print(f"    {k}: {v}")

    print("\n  Optimization level distribution:")
    opt_counts = {}
    for s in samples:
        opt_counts[s.opt_level] = opt_counts.get(s.opt_level, 0) + 1
    for k, v in sorted(opt_counts.items()):
        print(f"    {k}: {v}")

    return samples

# =============================================================================
# Prompt and Code Utilities
# =============================================================================

def build_prompt(ghidra_code: str) -> str:
    """Build LLM4Decompile prompt."""
    return f"# This is the assembly code:\n{ghidra_code}\n# What is the source code?\n"


def extract_c_code(text: str) -> str:
    """Extract C code from model output."""
    if "```c" in text:
        start = text.find("```c") + 4
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    return text.strip()


def extract_function_body(code: str) -> str:
    """Extract function body (remove includes)."""
    lines = [l for l in code.split('\n') if not l.strip().startswith('#include')]
    return '\n'.join(lines).strip()


# =============================================================================
# Compilation and Testing
# =============================================================================

def simplify_type(type_str):
    """Reduces complex C types to basic Python/JSON types."""
    type_str = type_str.lower()
    
    # 1. Handle Strings
    if 'string' in type_str:
        return 'string'
    
    # 2. Handle Booleans
    if 'bool' in type_str:
        return 'bool'
    
    # 3. Handle Floats (float32, float64, double)
    if 'float' in type_str:
        return 'float'
    
    # 4. Handle Integers (int8, int16, int32, uint32, etc.)
    if 'int' in type_str:
        return 'int'
    
    # 5. Fallback for pointers/structs - handle as strings or raw
    return 'string'

def check_compilation(args: Tuple[str, str, str]) -> bool:
    # print(1)
    code, c_func, c_test = args
    curr_dir = str(PROJECT_DIR)
    pattern = r'extern\s+"C"\s*\{[\s\S]*?#include\s+["<][^">]+[">][\s\S]*?\}'
    # print(code)
    # 2. Perform the replacement
    # final_content = re.sub(pattern, code.strip(), c_test)
    try:
        final_content = re.sub(pattern, lambda _: code.strip(), c_test)
    except re.error as e:
        # print(e)
        # regex replacement/parsing issue for this candidate
        return False
    except Exception as e:
        # print(e)
        return False
    # print(final_content)
    temp_dir = tempfile.mkdtemp(prefix="compile_", dir=curr_dir)
    try:
        c_file = os.path.join(temp_dir, 'test.cpp')
        exe = os.path.join(temp_dir, 'test')

        with open(c_file, 'w') as f:
            f.write(final_content)

        try:
            result = subprocess.run(
                f'g++ {c_file} -I . -o {exe}',
                shell=True, capture_output=True, timeout=10
            )
            # if result.returncode!=0:
            #     print(result.stderr.decode())
            # print(result.returncode)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            # print("compilation timeout")
            return False
        except Exception as e:
            # print(e)
            return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)



def run_test(args: Tuple[str, str, str, dict, str], config: Optional[Config] = None) -> bool:
    cfg = config or Config()
    code, c_func, c_test, real_io_pairs, real_iospec = args
    typemap = json.loads(real_iospec)['typemap']
    inputs_data = real_io_pairs['input']
    expected_outputs = real_io_pairs['output']
    curr_dir = str(PROJECT_DIR)
    all_match = True
    pattern = r'extern\s+"C"\s*\{[\s\S]*?#include\s+["<][^">]+[">][\s\S]*?\}'

    # 2. Perform the replacement
    try:
        final_content = re.sub(pattern, lambda _: code.strip(), c_test)
    except re.error:
        # regex replacement/parsing issue for this candidate
        return False
    except Exception:
        return False

    temp_dir = tempfile.mkdtemp(prefix="runtest_", dir=curr_dir)
    try:
        c_file = os.path.join(temp_dir, 'test.cpp')
        exe = os.path.join(temp_dir, 'test')
        input_json = os.path.join(temp_dir, 'input_tmp.json')
        output_json = os.path.join(temp_dir, 'output_tmp.json')

        with open(c_file, 'w') as f:
            f.write(final_content)

        try:
            result = subprocess.run(
                f'g++ -O0 {c_file} -I . -o {exe}',
                shell=True, capture_output=True, timeout=10
            )
            if result.returncode != 0:
                return False
        except Exception:
            return False

        for idx, (inp_dict, exp_out) in enumerate(zip(inputs_data, expected_outputs)):
            test_case_input = {}
            for var_name, raw_val in zip(inp_dict['var'], inp_dict['value']):
                raw_type = typemap.get(var_name, "string")
                base_type = simplify_type(raw_type)  # Using your simplify_type function

                try:
                    parsed_val = json.loads(raw_val)
                except Exception:
                    parsed_val = raw_val

                # Casting logic (same as yours)
                if base_type == "int":
                    test_case_input[var_name] = [int(x) for x in parsed_val] if isinstance(parsed_val, list) else int(parsed_val)
                elif base_type == "float":
                    test_case_input[var_name] = [float(x) for x in parsed_val] if isinstance(parsed_val, list) else float(parsed_val)
                elif base_type == "bool":
                    test_case_input[var_name] = bool(parsed_val)
                else:
                    test_case_input[var_name] = str(parsed_val)

            # 3. Save this SINGLE test case to input.json
            with open(input_json, "w") as f:
                json.dump(test_case_input, f, indent=4)

            # 2. Execute the validator
            # Command: ./validator input_tmp.json output_tmp.json
            try:
                result = subprocess.run(
                    f'{os.path.join(curr_dir, "run-sandboxed.sh")} ./test input_tmp.json output_tmp.json',
                    cwd=temp_dir,
                    shell=True,
                    capture_output=True,
                    timeout=20,
                )

                if result.returncode != 0:
                    all_match = False
                    break

                # 3. Load the generated output.json
                with open(output_json, "r") as f:
                    actual_output_json = json.load(f)

                # Iterate through every variable the dataset says we should expect
                for var_name, raw_expected in zip(exp_out['var'], exp_out['value']):
                    expected_val = json.loads(raw_expected)
                    actual_val = actual_output_json.get(var_name)

                    if actual_val != expected_val:
                        return False

            except Exception as e:
                print(e)
                all_match = False
                break

        return all_match
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Metrics
# =============================================================================

def compute_levenshtein(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    dist = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (dist / max_len)


def compute_codebleu(reference: str, candidate: str) -> float:
    """CodeBLEU score."""
    try:
        if not reference.strip() or not candidate.strip():
            return 0.0
        result = calc_codebleu(
            references=[[reference]],
            predictions=[candidate],
            lang="c",
            weights=(0.25, 0.25, 0.25, 0.25),
            tokenizer=None
        )
        return result['codebleu']
    except:
        return 0.0


# =============================================================================
# Perplexity Computation
# =============================================================================

class PerplexityComputer:
    """Compute perplexity for candidate selection."""

    def __init__(self, model_path: str, max_sequence_len: int = 4096):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_sequence_len = max_sequence_len
        print(f"Loading model for perplexity: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def compute(self, ghidra_code: str, decompiled_code: str) -> float:
        """Compute perplexity of decompiled code given Ghidra prompt."""
        try:
            prompt = build_prompt(ghidra_code)
            full_text = prompt + decompiled_code

            prompt_tokens = self.tokenizer(prompt, return_tensors="pt", truncation=False)
            full_tokens = self.tokenizer(full_text, return_tensors="pt", truncation=False)

            prompt_len = prompt_tokens.input_ids.shape[1]
            full_len = full_tokens.input_ids.shape[1]
            if full_len > self.max_sequence_len:
                return float('inf')
            decompiled_len = full_len - prompt_len

            if decompiled_len < 5:
                return float('inf')

            input_ids = full_tokens.input_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits

            target_logits = logits[0, prompt_len-1:full_len-1, :]
            target_tokens = input_ids[0, prompt_len:full_len]

            log_probs = F.log_softmax(target_logits, dim=-1)
            token_log_probs = log_probs.gather(1, target_tokens.unsqueeze(1)).squeeze(1)

            avg_log_prob = token_log_probs.float().mean().item()
            perplexity = float(torch.exp(torch.tensor(-avg_log_prob)).item())

            return perplexity
        except Exception as e:
            return float('inf')


# =============================================================================
# Candidate Generation with vLLM
# =============================================================================

def generate_candidates(
    model_path: str,
    samples: List[Sample],
    config: Config,
    cache_file: Path
) -> List[Sample]:
    """Generate candidates using vLLM."""

    incremental_cache = cache_file.with_name(cache_file.stem + "_incremental.pkl")

    if cache_file.exists():
        print(f"\nLoading cached candidates from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"\n=== Generating Candidates with vLLM ===")
    print(f"Model: {model_path}")
    print(f"Candidates per sample: {config.num_candidates}")
    print(f"Temperatures: {config.temperatures}")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype='bfloat16',
        max_model_len=config.max_sequence_len,
        gpu_memory_utilization=0.75,
        tensor_parallel_size=1,
    )

    tokenizer = llm.get_tokenizer()
    valid_samples = []
    skipped = 0
    prompt_lengths = {}
    available_tokens = {}
    for s in samples:
        token_len = compute_prompt_length(s.ghidra_code, tokenizer)
        prompt_lengths[s.idx] = token_len
        if token_len < config.max_sequence_len:
            valid_samples.append(s)
            available_tokens[s.idx] = config.max_sequence_len - token_len
        else:
            skipped += 1

    print(f"Valid samples: {len(valid_samples)}, Skipped (too long): {skipped}")

    candidates_per_temp = config.num_candidates // len(config.temperatures)
    sample_to_candidates = {s.idx: [] for s in valid_samples}

    completed_temp_indices = set()
    if incremental_cache.exists():
        print(f"\nLoading incremental candidate cache from {incremental_cache}")
        with open(incremental_cache, 'rb') as f:
            incremental_data = pickle.load(f)
        cached_candidates = incremental_data.get('sample_to_candidates', {})
        for key, value in cached_candidates.items():
            sample_to_candidates[key] = value
        completed_temp_indices = set(incremental_data.get('completed_temp_indices', []))
        print(f"  Completed temperatures: {sorted(completed_temp_indices)}")

    def save_incremental() -> None:
        with open(incremental_cache, 'wb') as f:
            pickle.dump(
                {
                    'sample_to_candidates': sample_to_candidates,
                    'completed_temp_indices': sorted(completed_temp_indices),
                },
                f,
            )

    def iter_budget_batches(sample_group: List[Sample], batch_size: int = 32):
        sorted_samples = sorted(sample_group, key=lambda s: available_tokens[s.idx], reverse=True)
        for start in range(0, len(sorted_samples), batch_size):
            batch = sorted_samples[start:start + batch_size]
            batch_budget = min(available_tokens[s.idx] for s in batch)
            yield batch, batch_budget

    for temp_idx, temp in enumerate(tqdm(config.temperatures, desc="Temperatures")):
        if temp_idx in completed_temp_indices:
            continue

        for batch_samples, batch_budget in tqdm(
            list(iter_budget_batches(valid_samples, batch_size=32)),
            desc=f"Temp {temp}",
            leave=False
        ):
            if batch_budget <= 0:
                continue

            params = SamplingParams(
                temperature=max(temp, 0.01),
                max_tokens=min(config.max_new_tokens, batch_budget),
                top_p=0.95 if temp > 0 else 1.0,
                n=candidates_per_temp,
            )

            prompts = [build_prompt(sample.ghidra_code) for sample in batch_samples]
            outputs = llm.generate(prompts, params)

            for sample, output in zip(batch_samples, outputs):
                for out in output.outputs:
                    code = extract_c_code(out.text)
                    sample_to_candidates[sample.idx].append(
                        Candidate(code=code, temperature=temp)
                    )

        completed_temp_indices.add(temp_idx)
        save_incremental()

    for sample in valid_samples:
        sample.candidates = sample_to_candidates[sample.idx]

    del llm
    torch.cuda.empty_cache()

    print(f"Saving candidates to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(samples, f)

    if incremental_cache.exists():
        incremental_cache.unlink()

    return samples


# =============================================================================
# Full Evaluation Pipeline
# =============================================================================

def evaluate_model(
    model_name: str,
    model_path: str,
    samples: List[Sample],
    config: Config,
) -> Dict:
    """Full evaluation pipeline for a single model."""

    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    model_key = model_name.replace(' ', '_').replace('/', '_')

    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")

    # Step 1: Generate candidates
    cand_cache = cache_dir / f"candidates_{model_key}.pkl"
    samples = generate_candidates(model_path, samples, config, cand_cache)

    # Step 2: Check compilation
    compile_cache = cache_dir / f"compilation_{model_key}.pkl"

    if compile_cache.exists():
        print("\nLoading cached compilation results...")
        with open(compile_cache, 'rb') as f:
            compile_results = pickle.load(f)
        for s in samples:
            for i, c in enumerate(s.candidates):
                key = (s.idx, i)
                if key in compile_results:
                    c.compiles = compile_results[key]
    else:
        print("\n=== Checking Compilation ===")
        tasks = []
        task_info = []
        for s in samples:
            for i, c in enumerate(s.candidates):
                tasks.append((c.code, s.c_func, s.c_test))
                task_info.append((s.idx, i))

        print(f"Checking {len(tasks)} candidates...")
        with mp.Pool(config.num_workers) as pool:
            results = list(tqdm(pool.imap(check_compilation, tasks), total=len(tasks)))

        compile_results = {}
        for (sample_idx, cand_idx), compiles in zip(task_info, results):
            compile_results[(sample_idx, cand_idx)] = compiles
            for s in samples:
                if s.idx == sample_idx:
                    s.candidates[cand_idx].compiles = compiles
                    break

        with open(compile_cache, 'wb') as f:
            pickle.dump(compile_results, f)

    total_compiling = sum(sum(1 for c in s.candidates if c.compiles) for s in samples)
    print(f"Total compiling candidates: {total_compiling}")

    # Step 3: Compute perplexity for compiling candidates
    ppl_cache = cache_dir / f"perplexity_{model_key}.pkl"

    if ppl_cache.exists():
        print("\nLoading cached perplexity results...")
        with open(ppl_cache, 'rb') as f:
            ppl_results = pickle.load(f)
        for s in samples:
            for i, c in enumerate(s.candidates):
                key = (s.idx, i)
                if key in ppl_results:
                    c.perplexity = ppl_results[key]
    else:
        print("\n=== Computing Perplexity ===")
        ppl_computer = PerplexityComputer(model_path, max_sequence_len=config.max_sequence_len)

        ppl_results = {}
        for s in tqdm(samples, desc="Perplexity"):
            for i, c in enumerate(s.candidates):
                if c.compiles:
                    ppl = ppl_computer.compute(s.ghidra_code, c.code)
                    c.perplexity = ppl
                    ppl_results[(s.idx, i)] = ppl

        del ppl_computer
        torch.cuda.empty_cache()

        with open(ppl_cache, 'wb') as f:
            pickle.dump(ppl_results, f)

    # Step 4: Select best candidate per sample (lowest perplexity)
    print("\n=== Selecting Best Candidates ===")
    for s in samples:
        compiling = [c for c in s.candidates if c.compiles]
        if compiling:
            best = min(compiling, key=lambda c: c.perplexity)
            s.selected_compiles = True

            orig_clean = extract_function_body(s.c_func)
            gen_clean = extract_function_body(best.code)

            best.levenshtein = compute_levenshtein(orig_clean, gen_clean)
            best.codebleu = compute_codebleu(orig_clean, gen_clean)

            s.selected_levenshtein = best.levenshtein
            s.selected_codebleu = best.codebleu

    # Step 5: Run tests on selected candidates
    test_cache = cache_dir / f"tests_{model_key}.pkl"

    if test_cache.exists():
        print("\nLoading cached test results...")
        with open(test_cache, 'rb') as f:
            test_results = pickle.load(f)
        for s in samples:
            if s.idx in test_results:
                s.selected_passes_tests = test_results[s.idx]
    else:
        print("\n=== Running Tests ===")
        test_tasks = []
        test_indices = []

        for s in samples:
            if s.selected_compiles:
                compiling = [c for c in s.candidates if c.compiles]
                best = min(compiling, key=lambda c: c.perplexity)
                test_tasks.append((best.code, s.c_func, s.c_test, s.real_io_pairs, s.real_iospec))
                test_indices.append(s.idx)

        print(f"Testing {len(test_tasks)} selected candidates...")
        # with mp.Pool(config.num_workers) as pool:
        #     results = list(tqdm(pool.imap(run_test, test_tasks), total=len(test_tasks)))

        # test_tasks = [(c, sample.c_func, sample.c_test, sample.real_io_pairs, sample.real_iospec) for c in compiling_candidates]
        with ThreadPoolExecutor(max_workers=max(1, 64)) as ex:
            results = list(tqdm(ex.map(lambda t: run_test(t, config), test_tasks),total=len(test_tasks)))

        test_results = {}
        for idx, passes in zip(test_indices, results):
            test_results[idx] = passes
            for s in samples:
                if s.idx == idx:
                    s.selected_passes_tests = passes
                    break

        with open(test_cache, 'wb') as f:
            pickle.dump(test_results, f)

    # Step 6: Compute statistics
    return compute_statistics(samples, model_name, config)


def compute_statistics(samples: List[Sample], model_name: str, config: Config) -> Dict:
    """Compute comprehensive statistics."""

    n_total = len(samples)
    n_compile = sum(1 for s in samples if s.selected_compiles)
    n_pass = sum(1 for s in samples if s.selected_passes_tests)

    stats = {
        'model': model_name,
        'config': {
            'num_candidates': config.num_candidates,
            'temperatures': config.temperatures,
            'selection_method': 'perplexity (lowest)',
        },
        'total': n_total,
        'compile': n_compile,
        'compile_rate': n_compile / n_total * 100 if n_total > 0 else 0,
        'pass': n_pass,
        'pass_rate': n_pass / n_total * 100 if n_total > 0 else 0,
        'levenshtein': sum(s.selected_levenshtein for s in samples) / n_total * 100 if n_total > 0 else 0,
        'codebleu': sum(s.selected_codebleu for s in samples) / n_total * 100 if n_total > 0 else 0,
    }

    # Oracle stats (any compiling candidate passes)
    oracle_pass = 0
    for s in samples:
        compiling = [c for c in s.candidates if c.compiles]
        if compiling:
            test_tasks = [(c.code, s.c_func, s.c_test, s.real_io_pairs, s.real_iospec) for c in compiling[:20]]
            for task in test_tasks:
                if run_test(task):
                    oracle_pass += 1
                    break
    stats['oracle_pass'] = oracle_pass
    stats['oracle_pass_rate'] = oracle_pass / n_total * 100 if n_total > 0 else 0

    # By obfuscation pass
    stats['by_obf_pass'] = {}
    obf_passes = set(s.obf_pass for s in samples)
    for obf in obf_passes:
        obf_samples = [s for s in samples if s.obf_pass == obf]
        n = len(obf_samples)
        stats['by_obf_pass'][obf] = {
            'total': n,
            'pass_rate': sum(1 for s in obf_samples if s.selected_passes_tests) / n * 100 if n > 0 else 0,
            'compile_rate': sum(1 for s in obf_samples if s.selected_compiles) / n * 100 if n > 0 else 0,
            'levenshtein': sum(s.selected_levenshtein for s in obf_samples) / n * 100 if n > 0 else 0,
            'codebleu': sum(s.selected_codebleu for s in obf_samples) / n * 100 if n > 0 else 0,
        }

    # By optimization level
    stats['by_opt_level'] = {}
    opt_levels = set(s.opt_level for s in samples)
    for opt in opt_levels:
        opt_samples = [s for s in samples if s.opt_level == opt]
        n = len(opt_samples)
        stats['by_opt_level'][opt] = {
            'total': n,
            'pass_rate': sum(1 for s in opt_samples if s.selected_passes_tests) / n * 100 if n > 0 else 0,
            'compile_rate': sum(1 for s in opt_samples if s.selected_compiles) / n * 100 if n > 0 else 0,
            'levenshtein': sum(s.selected_levenshtein for s in opt_samples) / n * 100 if n > 0 else 0,
            'codebleu': sum(s.selected_codebleu for s in opt_samples) / n * 100 if n > 0 else 0,
        }

    return stats


def print_statistics(stats: Dict):
    """Print formatted statistics."""
    print(f"\n{'='*70}")
    print(f"RESULTS: {stats['model']}")
    print(f"{'='*70}")

    print(f"\nSelection: {stats['config']['num_candidates']} candidates, {stats['config']['selection_method']}")

    print(f"\nOverall:")
    print(f"  Samples:      {stats['total']:,}")
    print(f"  Compile rate: {stats['compile']:,}/{stats['total']:,} = {stats['compile_rate']:.1f}%")
    print(f"  Pass@1:       {stats['pass']:,}/{stats['total']:,} = {stats['pass_rate']:.1f}%")
    print(f"  Levenshtein:  {stats['levenshtein']:.1f}%")
    print(f"  CodeBLEU:     {stats['codebleu']:.1f}%")

    print(f"\nBy Obfuscation Pass:")
    print(f"  {'Pass':<20} {'Pass@1':>8} {'Compile':>8} {'Lev':>8} {'CodeBLEU':>8} {'n':>6}")
    print(f"  {'-'*60}")
    for obf, data in sorted(stats['by_obf_pass'].items()):
        print(f"  {obf:<20} {data['pass_rate']:>7.1f}% {data['compile_rate']:>7.1f}% "
              f"{data['levenshtein']:>7.1f}% {data['codebleu']:>7.1f}% {data['total']:>6}")

    print(f"\nBy Optimization Level:")
    print(f"  {'Opt':<6} {'Pass@1':>8} {'Compile':>8} {'Lev':>8} {'CodeBLEU':>8} {'n':>6}")
    print(f"  {'-'*50}")
    for opt, data in sorted(stats['by_opt_level'].items()):
        print(f"  {opt:<6} {data['pass_rate']:>7.1f}% {data['compile_rate']:>7.1f}% "
              f"{data['levenshtein']:>7.1f}% {data['codebleu']:>7.1f}% {data['total']:>6}")


def print_comparison(stats_orig: Dict, stats_ft: Dict):
    """Print comparison between models."""
    print(f"\n{'='*70}")
    print("COMPARISON: ORIGINAL vs FINE-TUNED (with selection)")
    print(f"{'='*70}")

    print(f"\n{'Metric':<20} {'Original':>12} {'Fine-tuned':>12} {'Delta':>12}")
    print("-" * 58)

    for name, key in [('Pass@1 (%)', 'pass_rate'), ('Compile (%)', 'compile_rate'),
                      ('Levenshtein (%)', 'levenshtein'), ('CodeBLEU (%)', 'codebleu')]:
        orig = stats_orig[key]
        ft = stats_ft[key]
        delta = ft - orig
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        print(f"{name:<20} {orig:>11.1f}% {ft:>11.1f}% {delta_str:>11}pp")

    print(f"\nBy Obfuscation Pass (Pass@1):")
    print(f"  {'Obf Pass':<20} {'Original':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print(f"  {'-'*55}")
    for obf in sorted(stats_orig['by_obf_pass'].keys()):
        orig = stats_orig['by_obf_pass'][obf]['pass_rate']
        ft = stats_ft['by_obf_pass'][obf]['pass_rate']
        delta = ft - orig
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        print(f"  {obf:<20} {orig:>9.1f}% {ft:>11.1f}% {delta_str:>9}pp")


# =============================================================================
# Main
# =============================================================================

def main():
    config = Config()

    print("=" * 70)
    print("OBFUSCATED DECOMPILATION EVALUATION WITH SELECTION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Candidates per sample: {config.num_candidates}")
    print(f"  Temperatures: {config.temperatures}")
    print(f"  Selection method: Perplexity (lowest)")
    print(f"  Original model: {config.original_model}")
    print(f"  Fine-tuned model: {config.finetuned_model}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    samples = load_obfuscated_data(config)

    # Evaluate original model
    samples_orig = [Sample(
        idx=s.idx, obf_pass=s.obf_pass, opt_level=s.opt_level,
        ghidra_code=s.ghidra_code, c_func=s.c_func, c_test=s.c_test,real_io_pairs=s.real_io_pairs, real_iospec=s.real_iospec) for s in samples]

    stats_original = evaluate_model(
        "Original LLM4Decompile-9B",
        config.original_model,
        samples_orig,
        config
    )
    print_statistics(stats_original)

    # Evaluate fine-tuned model
    samples_ft = [Sample(
        idx=s.idx, obf_pass=s.obf_pass, opt_level=s.opt_level,
        ghidra_code=s.ghidra_code, c_func=s.c_func, c_test=s.c_test, real_io_pairs=s.real_io_pairs, real_iospec=s.real_iospec
    ) for s in samples]

    stats_finetuned = evaluate_model(
        "Fine-tuned LLM4Decompile-9B",
        config.finetuned_model,
        samples_ft,
        config
    )
    print_statistics(stats_finetuned)

    print_comparison(stats_original, stats_finetuned)

    # Save results
    summary = {
        'config': {
            'num_candidates': config.num_candidates,
            'temperatures': config.temperatures,
            'selection_method': 'perplexity (lowest)',
            'original_model': config.original_model,
            'finetuned_model': config.finetuned_model,
            'num_samples': len(samples),
        },
        'original': stats_original,
        'finetuned': stats_finetuned,
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Save LaTeX-friendly format
    latex_data = {
        'overall': {
            'metrics': ['Pass@1', 'Compile', 'Levenshtein', 'CodeBLEU'],
            'original': [
                stats_original['pass_rate'],
                stats_original['compile_rate'],
                stats_original['levenshtein'],
                stats_original['codebleu'],
            ],
            'finetuned': [
                stats_finetuned['pass_rate'],
                stats_finetuned['compile_rate'],
                stats_finetuned['levenshtein'],
                stats_finetuned['codebleu'],
            ],
        },
        'by_obf_pass': {},
        'by_opt_level': {},
    }

    for obf in sorted(stats_original['by_obf_pass'].keys()):
        latex_data['by_obf_pass'][obf] = {
            'original': stats_original['by_obf_pass'][obf],
            'finetuned': stats_finetuned['by_obf_pass'][obf],
        }

    for opt in sorted(stats_original['by_opt_level'].keys()):
        latex_data['by_opt_level'][opt] = {
            'original': stats_original['by_opt_level'][opt],
            'finetuned': stats_finetuned['by_opt_level'][opt],
        }

    with open(output_dir / "latex_data.json", 'w') as f:
        json.dump(latex_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to {output_dir}/")
    print(f"  - summary.json: Full results")
    print(f"  - latex_data.json: LaTeX-friendly format")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
