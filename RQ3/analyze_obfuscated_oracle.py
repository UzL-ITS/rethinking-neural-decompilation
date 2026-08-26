#!/usr/bin/env python3
"""
Analyze LLM4Decompile pass@256 performance on obfuscated code.

This script:
1. Loads the obfuscated dataset with different obfuscation types and optimization levels
2. Generates 256 decompiled candidates per sample using vLLM
3. Tests if any candidate compiles and passes functional tests (oracle/pass@256)
4. Saves detailed results for each (sample, obfuscation, opt_level) combination
5. Creates a summary by obfuscation type and optimization level

Results are saved to cache for subsequent analysis and plotting.
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import subprocess
import tempfile
import multiprocessing as mp
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import pickle
from pathlib import Path
import pandas as pd
import re
from vllm import LLM, SamplingParams
import shutil
import shlex
from concurrent.futures import ThreadPoolExecutor
import sys
from transformers import AutoTokenizer
PROJECT_DIR = Path(__file__).resolve().parent
_SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Data
    data_path: str = str(_SCRIPT_DIR / "../RQ3_obfuscation_finetuning/train-test-split_v2/test_non_empty_functions.json")

    # Model
    model_name: str = "LLM4Binary/llm4decompile-9b-v2"
    max_model_len: int = 4096

    # Generation
    num_candidates: int = 256
    temperatures: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    max_new_tokens: int = 4096

    # Execution
    num_workers: int = 96
    compile_timeout: int = 10
    run_timeout: int = 5

    # Cache
    cache_dir: str = str(_SCRIPT_DIR / "cache_oracle")

    # Limit for testing (None = all samples)
    max_samples: Optional[int] = None
    test_step_timeout: int = 10


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Sample:
    """A single sample from the obfuscated dataset."""
    idx: int
    task_id: str
    opt_level: str
    obf_pass: str
    ghidra: str
    c_func: str
    c_test: str
    func_name: str
    real_io_pairs: dict
    real_iospec: str


@dataclass
class SampleResult:
    """Results for a single sample."""
    idx: int
    task_id: str
    opt_level: str
    obf_pass: str
    num_candidates: int
    num_compiling: int
    num_passing: int
    oracle_pass: bool


# ============================================================================
# Utility Functions
# ============================================================================

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
                    timeout=cfg.test_step_timeout,
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
                # print(e)
                all_match = False
                break

        return all_match
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def create_prompt(ghidra_code: str) -> str:
    """Create prompt for LLM4Decompile."""
    return f"""# This is the assembly code:
{ghidra_code}
# What is the source code?
"""


def compute_prompt_length(ghidra_code: str, tokenizer) -> int:
    """Compute prompt token length for inference budgeting."""
    prompt = create_prompt(ghidra_code)
    return len(
        tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
            return_tensors=None,
        )["input_ids"]
    )


def compute_total_sequence_length(ghidra_code: str, source_code: str, tokenizer) -> int:
    """Compute full prompt+target sequence length before truncation."""
    prompt = create_prompt(ghidra_code)
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


# ============================================================================
# Main Pipeline
# ============================================================================

def load_dataset(config: Config) -> List[Sample]:
    """Load the obfuscated dataset."""
    print(f"Loading dataset from {config.data_path}")

    with open(config.data_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        model_max_length=config.max_model_len,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Filtering samples by full sequence length...")
    valid_indices = []
    skipped_long = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering lengths"):
        total_len = compute_total_sequence_length(
            row['input_asm_prompt'],
            row['c_func'],
            tokenizer,
        )
        if total_len <= config.max_model_len:
            valid_indices.append(idx)
        else:
            skipped_long += 1
    df = df.iloc[valid_indices].copy()
    print(f"  Skipped {skipped_long:,} overlength samples")
    print(f"  Samples within length limit: {len(df):,}")

    df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    print(f"No. of OPT levels: {df['type'].value_counts()}")
    print(f"No. of obf passes {df['obf_pass'].astype(str).value_counts()}")
    # df=df[:300]
    samples = []
    for idx, row in df.iterrows():
        obf_pass = ','.join(row['obf_pass']) if isinstance(row['obf_pass'], list) else str(row['obf_pass'])

        sample = Sample(
            idx=idx,
            task_id=row['task_id'],
            opt_level=row['type'],
            obf_pass=obf_pass,
            ghidra=row['input_asm_prompt'],
            c_func=row['c_func'],
            c_test=row['c_test'],
            func_name=row['func_name'],
            real_io_pairs=row['real_io_pairs'],
            real_iospec=row['real_iospec']
        )
        samples.append(sample)

    if config.max_samples:
        samples = samples[:config.max_samples]

    print(f"Loaded {len(samples)} samples")
    return samples


def generate_candidates(
    samples: List[Sample],
    config: Config,
    cache_dir: Path
) -> Dict[int, List[str]]:
    """Generate candidates for all samples using vLLM."""

    cache_file = cache_dir / f"candidates_{len(samples)}_{config.num_candidates}.pkl"
    incremental_cache = cache_dir / f"candidates_{len(samples)}_{config.num_candidates}_incremental.pkl"

    if cache_file.exists():
        print(f"Loading cached candidates from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"\n{'='*70}")
    print("GENERATING CANDIDATES WITH vLLM")
    print(f"{'='*70}")
    print(f"Model: {config.model_name}")
    print(f"Samples: {len(samples)}")
    print(f"Candidates per sample: {config.num_candidates}")
    print(f"Temperatures: {config.temperatures}")

    llm = LLM(
        model=config.model_name,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=0.5
    )

    tokenizer = llm.get_tokenizer()
    print(f"\nPrompt-aware generation budget with max_model_len={config.max_model_len}")
    valid_samples = []
    skipped_samples = []
    prompt_lengths = {}
    available_tokens = {}
    for s in samples:
        token_len = compute_prompt_length(s.ghidra, tokenizer)
        prompt_lengths[s.idx] = token_len
        if token_len < config.max_model_len:
            valid_samples.append(s)
            available_tokens[s.idx] = config.max_model_len - token_len
        else:
            skipped_samples.append((s, token_len))

    print(f"  Valid samples: {len(valid_samples)}")
    print(f"  Skipped (prompt too long): {len(skipped_samples)}")
    if skipped_samples:
        longest = max(skipped_samples, key=lambda x: x[1])
        print(f"  Longest skipped: {longest[1]} tokens (sample idx={longest[0].idx})")

    candidates_per_temp = config.num_candidates // len(config.temperatures)
    extra = config.num_candidates % len(config.temperatures)

    all_candidates = {s.idx: [] for s in samples}
    completed_temp_indices = set()

    if incremental_cache.exists():
        print(f"Loading incremental candidate cache from {incremental_cache}")
        with open(incremental_cache, 'rb') as f:
            incremental_data = pickle.load(f)
        all_candidates.update(incremental_data.get('all_candidates', {}))
        completed_temp_indices = set(incremental_data.get('completed_temp_indices', []))
        print(f"  Completed temperatures: {sorted(completed_temp_indices)}")

    def save_incremental() -> None:
        with open(incremental_cache, 'wb') as f:
            pickle.dump(
                {
                    'all_candidates': all_candidates,
                    'completed_temp_indices': sorted(completed_temp_indices),
                },
                f,
            )

    def iter_budget_batches(sample_group: List[Sample], batch_size: int = 32) -> List[Tuple[List[Sample], int]]:
        sorted_samples = sorted(sample_group, key=lambda s: available_tokens[s.idx], reverse=True)
        batches = []
        for start in range(0, len(sorted_samples), batch_size):
            batch = sorted_samples[start:start + batch_size]
            batch_budget = min(available_tokens[s.idx] for s in batch)
            batches.append((batch, batch_budget))
        return batches

    for temp_idx, temperature in enumerate(config.temperatures):
        if temp_idx in completed_temp_indices:
            print(f"\nSkipping cached temperature {temperature}")
            continue

        n_candidates = candidates_per_temp + (1 if temp_idx < extra else 0)

        print(f"\nGenerating {n_candidates} candidates at temperature {temperature}")

        budget_batches = iter_budget_batches(valid_samples, batch_size=32)
        for batch_samples, batch_budget in tqdm(
            budget_batches,
            desc=f"Temp {temperature}",
            leave=False
        ):
            if batch_budget <= 0:
                continue

            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=min(config.max_new_tokens, batch_budget),
                n=n_candidates,
                stop=["# This is the assembly code:"]
            )

            prompts = [create_prompt(sample.ghidra) for sample in batch_samples]
            outputs = llm.generate(prompts, sampling_params)
            for sample, output in zip(batch_samples, outputs):
                for completion in output.outputs:
                    all_candidates[sample.idx].append(completion.text.strip())

        completed_temp_indices.add(temp_idx)
        save_incremental()
        generated_so_far = sum(len(v) for v in all_candidates.values())
        print(f"  Saved incremental candidates after temperature {temperature} "
              f"({generated_so_far} completions total)")

    with open(cache_file, 'wb') as f:
        pickle.dump(all_candidates, f)
    print(f"\nSaved candidates to {cache_file}")

    if incremental_cache.exists():
        incremental_cache.unlink()

    skipped_file = cache_dir / "skipped_samples.json"
    with open(skipped_file, 'w') as f:
        json.dump([{'idx': s.idx, 'task_id': s.task_id, 'obf_pass': s.obf_pass,
                    'opt_level': s.opt_level, 'token_len': tl}
                   for s, tl in skipped_samples], f, indent=2)
    print(f"Saved skipped samples info to {skipped_file}")

    return all_candidates


def evaluate_samples(
    samples: List[Sample],
    candidates: Dict[int, List[str]],
    config: Config,
    cache_dir: Path
) -> List[SampleResult]:
    """Evaluate all samples for pass@256 (oracle) with incremental caching."""

    results_cache = cache_dir / "sample_results.pkl"
    incremental_cache = cache_dir / "sample_results_incremental.pkl"

    if results_cache.exists():
        print(f"Loading cached results from {results_cache}")
        with open(results_cache, 'rb') as f:
            return pickle.load(f)

    evaluated_indices = set()
    results = []

    if incremental_cache.exists():
        print(f"Loading partial results from {incremental_cache}")
        with open(incremental_cache, 'rb') as f:
            results = pickle.load(f)
            evaluated_indices = {r.idx for r in results}
        print(f"  Found {len(results)} previously evaluated samples")

    remaining_samples = [s for s in samples if s.idx not in evaluated_indices]

    if not remaining_samples:
        print("All samples already evaluated!")
        with open(results_cache, 'wb') as f:
            pickle.dump(results, f)
        return results

    print(f"\n{'='*70}")
    print("EVALUATING SAMPLES")
    print(f"{'='*70}")
    print(f"  Total samples: {len(samples)}")
    print(f"  Already evaluated: {len(evaluated_indices)}")
    print(f"  Remaining: {len(remaining_samples)}")

    save_interval = 50

    for i, sample in enumerate(tqdm(remaining_samples, desc="Evaluating")):
        sample_candidates = candidates.get(sample.idx, [])

        if not sample_candidates:
            results.append(SampleResult(
                idx=sample.idx,
                task_id=sample.task_id,
                opt_level=sample.opt_level,
                obf_pass=sample.obf_pass,
                num_candidates=0,
                num_compiling=0,
                num_passing=0,
                oracle_pass=False
            ))
            continue

        compile_tasks = [(c, sample.c_func, sample.c_test) for c in sample_candidates]

        with mp.Pool(config.num_workers) as pool:
            compile_results = pool.map(check_compilation, compile_tasks)

        compiling_candidates = [c for c, compiles in zip(sample_candidates, compile_results) if compiles]
        num_compiling = len(compiling_candidates)

        num_passing = 0
        oracle_pass = False

        if compiling_candidates:
#             test_tasks = [(c, sample.c_func, sample.c_test) for c in compiling_candidates]

#             with mp.Pool(config.num_workers) as pool:
#                 test_results = pool.map(run_test, test_tasks)
                
            test_tasks = [(c, sample.c_func, sample.c_test, sample.real_io_pairs, sample.real_iospec) for c in compiling_candidates]
            with ThreadPoolExecutor(max_workers=max(1, 64)) as ex:
                test_results = list(ex.map(lambda t: run_test(t, config), test_tasks))
                
            num_passing = sum(test_results)
            oracle_pass = num_passing > 0

        results.append(SampleResult(
            idx=sample.idx,
            task_id=sample.task_id,
            opt_level=sample.opt_level,
            obf_pass=sample.obf_pass,
            num_candidates=len(sample_candidates),
            num_compiling=num_compiling,
            num_passing=num_passing,
            oracle_pass=oracle_pass
        ))

        if (i + 1) % save_interval == 0:
            with open(incremental_cache, 'wb') as f:
                pickle.dump(results, f)
            oracle_count = sum(1 for r in results if r.oracle_pass)
            print(f"\n  [Checkpoint] {len(results)}/{len(samples)} evaluated, "
                  f"oracle pass rate: {100*oracle_count/len(results):.1f}%")

    with open(results_cache, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved final results to {results_cache}")

    if incremental_cache.exists():
        incremental_cache.unlink()

    return results


def print_summary(results: List[SampleResult], cache_dir: Path):
    """Print and save summary statistics."""

    print(f"\n{'='*70}")
    print("SUMMARY: PASS@256 (ORACLE) BY OBFUSCATION AND OPTIMIZATION")
    print(f"{'='*70}")

    df = pd.DataFrame([asdict(r) for r in results])

    total = len(df)
    oracle_pass_total = df['oracle_pass'].sum()
    print(f"\nOverall: {oracle_pass_total}/{total} = {100*oracle_pass_total/total:.1f}%")

    print("\nBy Obfuscation Type:")
    for obf in sorted(df['obf_pass'].unique()):
        subset = df[df['obf_pass'] == obf]
        passed = subset['oracle_pass'].sum()
        total_obf = len(subset)
        print(f"  {obf:15s}: {passed:3d}/{total_obf:3d} = {100*passed/total_obf:5.1f}%")

    print("\nBy Optimization Level:")
    for opt in ['O0', 'O1', 'O2', 'O3']:
        subset = df[df['opt_level'] == opt]
        passed = subset['oracle_pass'].sum()
        total_opt = len(subset)
        pct = float(100 * passed / total_opt) if total_opt > 0 else 0.0 # Solving divide by infinity error
        # print(f"  {opt}: {passed:3d}/{total_opt:3d} = {100*passed/total_opt:5.1f}%")
        print(f"  {opt}: {passed:3d}/{total_opt:3d} = {pct:5.1f}%")

    print("\nHeatmap (rows=obfuscation, cols=optimization):")

    obf_types = sorted(df['obf_pass'].unique())
    opt_levels = ['O0', 'O1', 'O2', 'O3']

    heatmap_data = {}
    for obf in obf_types:
        heatmap_data[obf] = {}
        for opt in opt_levels:
            subset = df[(df['obf_pass'] == obf) & (df['opt_level'] == opt)]
            if len(subset) > 0:
                passed = subset['oracle_pass'].sum()
                total_cell = len(subset)
                heatmap_data[obf][opt] = {
                    'passed': int(passed),
                    'total': int(total_cell),
                    'percentage': float(100 * passed / total_cell)
                }

    print(f"\n{'':15s}", end='')
    for opt in opt_levels:
        print(f"{opt:>10s}", end='')
    print()

    for obf in obf_types:
        print(f"{obf:15s}", end='')
        for opt in opt_levels:
            if opt in heatmap_data[obf]:
                pct = heatmap_data[obf][opt]['percentage']
                print(f"{pct:9.1f}%", end='')
            else:
                print(f"{'N/A':>10s}", end='')
        print()

    results_json = {
        'overall': {
            'passed': int(oracle_pass_total),
            'total': int(total),
            'percentage': float(100 * oracle_pass_total / total)
        },
        'by_obfuscation': {},
        'by_optimization': {},
        'heatmap': heatmap_data,
        'detailed_results': [asdict(r) for r in results]
    }

    for obf in obf_types:
        subset = df[df['obf_pass'] == obf]
        passed = subset['oracle_pass'].sum()
        results_json['by_obfuscation'][obf] = {
            'passed': int(passed),
            'total': int(len(subset)),
            'percentage': float(100 * passed / len(subset))
        }

    for opt in opt_levels:
        subset = df[df['opt_level'] == opt]
        passed = subset['oracle_pass'].sum()
        total_subset = len(subset)
        pct = float(100 * passed / total_subset) if total_subset > 0 else 0.0
        results_json['by_optimization'][opt] = {
            'passed': int(passed),
            'total': int(total_subset),
            'percentage': pct
        }

    results_file = cache_dir / "oracle_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return results_json


def main():
    config = Config()
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("LLM4DECOMPILE PASS@256 ANALYSIS ON OBFUSCATED CODE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Candidates per sample: {config.num_candidates}")
    print(f"  Temperatures: {config.temperatures}")
    print(f"  Cache directory: {config.cache_dir}")

    samples = load_dataset(config)

    df = pd.DataFrame([{'obf': s.obf_pass, 'opt': s.opt_level} for s in samples])
    print(f"\nDataset composition:")
    print(df.groupby(['obf', 'opt']).size().unstack(fill_value=0))

    candidates = generate_candidates(samples, config, cache_dir)
    results = evaluate_samples(samples, candidates, config, cache_dir)
    summary = print_summary(results, cache_dir)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nTo visualize results, run: python plot_obfuscated_results.py")


if __name__ == "__main__":
    main()
