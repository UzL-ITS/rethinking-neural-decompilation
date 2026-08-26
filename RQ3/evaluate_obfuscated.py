#!/usr/bin/env python3
"""
Evaluate LLM4Decompile on obfuscated samples.

This script evaluates both:
1. Original LLM4Binary/llm4decompile-9b-v2 model
2. Fine-tuned model on obfuscation data

Metrics:
- Pass@1 (functional correctness)
- Compilation rate
- Levenshtein similarity
- CodeBLEU score

Usage:
    python evaluate_obfuscated.py
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from eval_utils import (
    build_llm4decompile_prompt,
    extract_c_code,
    compute_all_metrics,
)

_SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    # Data paths
    obfuscated_data: str = str(_SCRIPT_DIR / "../RQ3_obfuscation_finetuning/train-test-split_v2/test_non_empty_functions.json")
    selected_samples_file: str = str(_SCRIPT_DIR / "cache_oracle/oracle_results.json")

    # Models to evaluate
    original_model: str = "LLM4Binary/llm4decompile-9b-v2"
    finetuned_model: str = str(_SCRIPT_DIR / "checkpoints/llm4decompile-9b-obf-epoch1")

    # Generation settings
    max_new_tokens: int = 4096
    max_sequence_len: int = 4096
    num_samples: Optional[int] = None  # None = all samples

    # Output
    output_dir: str = str(_SCRIPT_DIR / "results_greedy")
    save_interval: int = 100


# =============================================================================
# Data Loading
# =============================================================================
def compute_total_sequence_length(ghidra_code: str, source_code: str, tokenizer) -> int:
    """Compute full prompt+target sequence length before truncation."""
    prompt = build_llm4decompile_prompt(ghidra_code)
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


def load_obfuscated_data(config: Config) -> List[Dict]:
    """Load obfuscated test data and convert to list format."""
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
        sampled_ids = df["task_id"].tolist()
        with open("sampled_ids_1000.txt", "w") as f:
            for x in sampled_ids:
                f.write(f"{x}\n")
    samples = []
    if isinstance(raw_data, list):
        num_samples = len(raw_data)
        print(f"  Found {num_samples} samples")

        for idx, row in enumerate(raw_data):
            sample = {
                'idx': idx,
                'func_name': row.get('func_name'),
                'type': row.get('type', row.get('opt_level')),
                'task_id': row.get('task_id', idx),
                'c_func': row['c_func'],
                'c_test': row['c_test'],
                'ghidra_code': row.get('input_asm_prompt', row.get('c_code', row.get('c_code_obf'))),
                'obf_pass': row['obf_pass'],
                'real_io_pairs': row['real_io_pairs'],
                'real_iospec': row['real_iospec']
            }
            samples.append(sample)
    elif isinstance(raw_data, dict):
        num_samples = len(raw_data['func_name'])
        print(f"  Found {num_samples} samples")

        for idx in range(num_samples):
            str_idx = str(idx)
            sample = {
                'idx': idx,
                'func_name': raw_data['func_name'][str_idx],
                'type': raw_data['type'][str_idx],
                'task_id': raw_data.get('task_id', {}).get(str_idx, idx),
                'c_func': raw_data['c_func'][str_idx],
                'c_test': raw_data['c_test'][str_idx],
                'ghidra_code': raw_data.get('input_asm_prompt', raw_data.get('c_code', raw_data.get('c_code_obf')))[str_idx],
                'obf_pass': raw_data['obf_pass'][str_idx],
                'real_io_pairs': raw_data['real_io_pairs'][str_idx],
                'real_iospec': raw_data['real_iospec'][str_idx]
            }
            samples.append(sample)
    else:
        raise TypeError(f"Unsupported JSON structure: {type(raw_data).__name__}")

    if config.num_samples is not None:
        samples = samples[:config.num_samples]
        print(f"  Using first {len(samples)} samples")

    print("\n  Obfuscation pass distribution:")
    obf_counts = {}
    for s in samples:
        key = str(s['obf_pass'])
        obf_counts[key] = obf_counts.get(key, 0) + 1
    for k, v in sorted(obf_counts.items()):
        print(f"    {k}: {v}")

    print("\n  Optimization level distribution:")
    opt_counts = {}
    for s in samples:
        opt_counts[s['type']] = opt_counts.get(s['type'], 0) + 1
    for k, v in sorted(opt_counts.items()):
        print(f"    {k}: {v}")

    return samples


# =============================================================================
# Model Loading and Generation
# =============================================================================
class DecompileModel:
    """Wrapper for LLM4Decompile model."""

    def __init__(self, model_path: str, name: str = "model", max_sequence_len: int = 4096):
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_sequence_len = max_sequence_len

        print(f"\nLoading {name} from {model_path}...")
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
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"  Model loaded: {self.model.num_parameters():,} parameters")

    def generate(self, ghidra_code: str, max_new_tokens: int = 2048) -> str:
        """Generate decompiled C code from Ghidra pseudo-code."""
        prompt = build_llm4decompile_prompt(ghidra_code)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]
        available_tokens = self.max_sequence_len - prompt_len
        generation_budget = min(max_new_tokens, available_tokens)
        if generation_budget <= 0:
            raise ValueError(f"Prompt length {prompt_len} leaves no room for generation")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=generation_budget,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs.attention_mask,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return extract_c_code(generated)


# =============================================================================
# Evaluation
# =============================================================================
@dataclass
class EvalResult:
    """Result for a single sample."""
    idx: int
    obf_pass: str
    opt_level: str
    generated_code: str
    compiles: bool
    passes_tests: bool
    levenshtein: float
    codebleu: Optional[float]
    bad_codebleu: bool
    generation_time: float


def serialize_result(result: EvalResult) -> Dict:
    return {
        'idx': result.idx,
        'obf_pass': result.obf_pass,
        'opt_level': result.opt_level,
        'generated_code': result.generated_code,
        'compiles': result.compiles,
        'passes_tests': result.passes_tests,
        'levenshtein': result.levenshtein,
        'codebleu': result.codebleu,
        'bad_codebleu': result.bad_codebleu,
        'generation_time': result.generation_time,
    }


def deserialize_result(data: Dict) -> EvalResult:
    return EvalResult(
        idx=data['idx'],
        obf_pass=data['obf_pass'],
        opt_level=data['opt_level'],
        generated_code=data.get('generated_code', ""),
        compiles=data['compiles'],
        passes_tests=data['passes_tests'],
        levenshtein=data['levenshtein'],
        codebleu=data.get('codebleu'),
        bad_codebleu=data.get('bad_codebleu', False),
        generation_time=data.get('generation_time', 0.0),
    )


def evaluate_model(
    model: DecompileModel,
    samples: List[Dict],
    config: Config,
    cache_path: Path,
) -> tuple[List[EvalResult], List[Dict]]:
    """Evaluate a model on all samples."""
    results = []
    bad_codebleu_samples = []
    evaluated_indices = set()

    if cache_path.exists():
        print(f"Loading incremental cache from {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        results = [deserialize_result(item) for item in cached.get('results', [])]
        bad_codebleu_samples = cached.get('bad_codebleu_samples', [])
        evaluated_indices = {r.idx for r in results}
        print(f"  Found {len(results)} cached samples")

    print(f"\nEvaluating {model.name}...")
    print("-" * 70)

    remaining_samples = [sample for sample in samples if sample['idx'] not in evaluated_indices]
    for i, sample in enumerate(tqdm(remaining_samples, desc=f"Evaluating {model.name}"), start=1):
        try:
            start_time = time.time()
            generated_code = model.generate(
                sample['ghidra_code'],
                max_new_tokens=config.max_new_tokens
            )
            gen_time = time.time() - start_time

            metrics = compute_all_metrics(
                generated_code=generated_code,
                original_code=sample['c_func'],
                c_func=sample['c_func'],
                c_test=sample['c_test'],
                real_io_pairs=sample['real_io_pairs'],
                real_iospec=sample['real_iospec'],
            )

            result = EvalResult(
                idx=sample['idx'],
                obf_pass=str(sample['obf_pass']),
                opt_level=sample['type'],
                generated_code=generated_code,
                compiles=metrics['compiles'],
                passes_tests=metrics['passes_tests'],
                levenshtein=metrics['levenshtein'],
                codebleu=metrics['codebleu'],
                bad_codebleu=metrics.get('bad_codebleu', False),
                generation_time=gen_time,
            )

            if result.bad_codebleu:
                bad_codebleu_samples.append(
                    {
                        'idx': sample['idx'],
                        'task_id': sample.get('task_id'),
                        'opt_level': sample.get('type'),
                        'obf_pass': sample.get('obf_pass'),
                        'func_name': sample.get('func_name'),
                        'reference_code': sample.get('c_func'),
                    }
                )

        except Exception as e:
            print(f"\nError on sample {sample['idx']}: {e}")
            result = EvalResult(
                idx=sample['idx'],
                obf_pass=str(sample['obf_pass']),
                opt_level=sample['type'],
                generated_code="",
                compiles=False,
                passes_tests=False,
                levenshtein=0.0,
                codebleu=None,
                bad_codebleu=True,
                generation_time=0.0,
            )

        results.append(result)

        if i % config.save_interval == 0:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'results': [serialize_result(r) for r in results],
                        'bad_codebleu_samples': bad_codebleu_samples,
                    },
                    f,
                    indent=2,
                )

    return results, bad_codebleu_samples


def compute_statistics(results: List[EvalResult], name: str) -> Dict:
    """Compute aggregate statistics from results."""
    n_total = len(results)
    n_compile = sum(1 for r in results if r.compiles)
    n_pass = sum(1 for r in results if r.passes_tests)
    avg_lev = sum(r.levenshtein for r in results) / n_total * 100
    valid_codebleu = [r.codebleu for r in results if r.codebleu is not None]
    print(f"Valid codebleus = {len(valid_codebleu)}")
    avg_cb = (sum(valid_codebleu) / len(valid_codebleu) * 100) if valid_codebleu else 0.0
    avg_time = sum(r.generation_time for r in results) / n_total

    stats = {
        'model': name,
        'total': n_total,
        'compile': n_compile,
        'compile_rate': n_compile / n_total * 100,
        'pass': n_pass,
        'pass_rate': n_pass / n_total * 100,
        'levenshtein': avg_lev,
        'codebleu': avg_cb,
        'codebleu_count': len(valid_codebleu),
        'avg_time': avg_time,
    }

    stats['by_obf_pass'] = {}
    obf_passes = set(r.obf_pass for r in results)
    for obf in obf_passes:
        obf_results = [r for r in results if r.obf_pass == obf]
        n = len(obf_results)
        cb_vals = [r.codebleu for r in obf_results if r.codebleu is not None]
        stats['by_obf_pass'][obf] = {
            'total': n,
            'pass_rate': sum(1 for r in obf_results if r.passes_tests) / n * 100,
            'compile_rate': sum(1 for r in obf_results if r.compiles) / n * 100,
            'levenshtein': sum(r.levenshtein for r in obf_results) / n * 100,
            'codebleu': (sum(cb_vals) / len(cb_vals) * 100) if cb_vals else 0.0,
        }

    stats['by_opt_level'] = {}
    opt_levels = set(r.opt_level for r in results)
    for opt in opt_levels:
        opt_results = [r for r in results if r.opt_level == opt]
        n = len(opt_results)
        cb_vals = [r.codebleu for r in opt_results if r.codebleu is not None]
        stats['by_opt_level'][opt] = {
            'total': n,
            'pass_rate': sum(1 for r in opt_results if r.passes_tests) / n * 100,
            'compile_rate': sum(1 for r in opt_results if r.compiles) / n * 100,
            'levenshtein': sum(r.levenshtein for r in opt_results) / n * 100,
            'codebleu': (sum(cb_vals) / len(cb_vals) * 100) if cb_vals else 0.0,
        }

    return stats


def print_statistics(stats: Dict):
    """Print formatted statistics."""
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {stats['model']}")
    print(f"{'=' * 70}")

    print(f"\nOverall:")
    print(f"  Samples:      {stats['total']:,}")
    print(f"  Compile rate: {stats['compile']:,}/{stats['total']:,} = {stats['compile_rate']:.1f}%")
    print(f"  Pass@1:       {stats['pass']:,}/{stats['total']:,} = {stats['pass_rate']:.1f}%")
    print(f"  Levenshtein:  {stats['levenshtein']:.1f}%")
    print(f"  CodeBLEU:     {stats['codebleu']:.1f}%")
    print(f"  CodeBLEU n:   {stats['codebleu_count']:,}/{stats['total']:,}")
    print(f"  Avg time:     {stats['avg_time']:.2f}s/sample")

    print(f"\nBy Obfuscation Pass:")
    print(f"  {'Pass':<20} {'Pass@1':>8} {'Compile':>8} {'Lev':>8} {'CodeBLEU':>8} {'n':>6}")
    print(f"  {'-'*60}")
    for obf, data in sorted(stats['by_obf_pass'].items()):
        print(f"  {obf:<20} {data['pass_rate']:>7.1f}% {data['compile_rate']:>7.1f}% {data['levenshtein']:>7.1f}% {data['codebleu']:>7.1f}% {data['total']:>6}")

    print(f"\nBy Optimization Level:")
    print(f"  {'Opt':<6} {'Pass@1':>8} {'Compile':>8} {'Lev':>8} {'CodeBLEU':>8} {'n':>6}")
    print(f"  {'-'*50}")
    for opt, data in sorted(stats['by_opt_level'].items()):
        print(f"  {opt:<6} {data['pass_rate']:>7.1f}% {data['compile_rate']:>7.1f}% {data['levenshtein']:>7.1f}% {data['codebleu']:>7.1f}% {data['total']:>6}")


def print_comparison(stats_original: Dict, stats_finetuned: Dict):
    """Print comparison between original and fine-tuned models."""
    print(f"\n{'=' * 70}")
    print("COMPARISON: ORIGINAL vs FINE-TUNED")
    print(f"{'=' * 70}")

    print(f"\n{'Metric':<20} {'Original':>12} {'Fine-tuned':>12} {'Delta':>12}")
    print("-" * 58)

    metrics = [
        ('Pass@1 (%)', 'pass_rate'),
        ('Compile (%)', 'compile_rate'),
        ('Levenshtein (%)', 'levenshtein'),
        ('CodeBLEU (%)', 'codebleu'),
    ]

    for name, key in metrics:
        orig = stats_original[key]
        ft = stats_finetuned[key]
        delta = ft - orig
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        print(f"{name:<20} {orig:>11.1f}% {ft:>11.1f}% {delta_str:>11}pp")

    print(f"\nBy Obfuscation Pass (Original -> Fine-tuned):")
    print(f"{'Obf Pass':<15} {'Pass@1':^25} {'Levenshtein':^25} {'CodeBLEU':^25}")
    print(f"{'':15} {'Orig':>8} {'FT':>8} {'D':>7} {'Orig':>8} {'FT':>8} {'D':>7} {'Orig':>8} {'FT':>8} {'D':>7}")
    print("-" * 95)

    for obf in sorted(stats_original['by_obf_pass'].keys()):
        orig_data = stats_original['by_obf_pass'][obf]
        ft_data = stats_finetuned['by_obf_pass'][obf]

        pass_delta = ft_data['pass_rate'] - orig_data['pass_rate']
        lev_delta = ft_data['levenshtein'] - orig_data['levenshtein']
        cb_delta = ft_data['codebleu'] - orig_data['codebleu']

        pass_d = f"+{pass_delta:.1f}" if pass_delta >= 0 else f"{pass_delta:.1f}"
        lev_d = f"+{lev_delta:.1f}" if lev_delta >= 0 else f"{lev_delta:.1f}"
        cb_d = f"+{cb_delta:.1f}" if cb_delta >= 0 else f"{cb_delta:.1f}"

        print(f"{obf:<15} {orig_data['pass_rate']:>7.1f}% {ft_data['pass_rate']:>7.1f}% {pass_d:>6} "
              f"{orig_data['levenshtein']:>7.1f}% {ft_data['levenshtein']:>7.1f}% {lev_d:>6} "
              f"{orig_data['codebleu']:>7.1f}% {ft_data['codebleu']:>7.1f}% {cb_d:>6}")

    print(f"\nBy Optimization Level (Original -> Fine-tuned):")
    print(f"{'Opt':<6} {'Pass@1':^25} {'Levenshtein':^25} {'CodeBLEU':^25}")
    print(f"{'':6} {'Orig':>8} {'FT':>8} {'D':>7} {'Orig':>8} {'FT':>8} {'D':>7} {'Orig':>8} {'FT':>8} {'D':>7}")
    print("-" * 86)

    for opt in sorted(stats_original['by_opt_level'].keys()):
        orig_data = stats_original['by_opt_level'][opt]
        ft_data = stats_finetuned['by_opt_level'][opt]

        pass_delta = ft_data['pass_rate'] - orig_data['pass_rate']
        lev_delta = ft_data['levenshtein'] - orig_data['levenshtein']
        cb_delta = ft_data['codebleu'] - orig_data['codebleu']

        pass_d = f"+{pass_delta:.1f}" if pass_delta >= 0 else f"{pass_delta:.1f}"
        lev_d = f"+{lev_delta:.1f}" if lev_delta >= 0 else f"{lev_delta:.1f}"
        cb_d = f"+{cb_delta:.1f}" if cb_delta >= 0 else f"{cb_delta:.1f}"

        print(f"{opt:<6} {orig_data['pass_rate']:>7.1f}% {ft_data['pass_rate']:>7.1f}% {pass_d:>6} "
              f"{orig_data['levenshtein']:>7.1f}% {ft_data['levenshtein']:>7.1f}% {lev_d:>6} "
              f"{orig_data['codebleu']:>7.1f}% {ft_data['codebleu']:>7.1f}% {cb_d:>6}")


# =============================================================================
# Main
# =============================================================================
def main():
    config = Config()

    print("=" * 70)
    print("OBFUSCATED DECOMPILATION EVALUATION")
    print("=" * 70)
    print(f"\nOriginal model:  {config.original_model}")
    print(f"Fine-tuned model: {config.finetuned_model}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    original_cache_path = output_dir / "incremental_original.json"
    finetuned_cache_path = output_dir / "incremental_finetuned.json"

    samples = load_obfuscated_data(config)

    model_original = DecompileModel(
        config.original_model,
        "Original LLM4Decompile-9B",
        max_sequence_len=config.max_sequence_len,
    )

    results_original, bad_codebleu_original = evaluate_model(model_original, samples, config, original_cache_path)
    stats_original = compute_statistics(results_original, "Original LLM4Decompile-9B")
    print_statistics(stats_original)

    with open(output_dir / "results_original.json", 'w') as f:
        json.dump({
            'stats': stats_original,
            'results': [
                {
                    'idx': r.idx,
                    'obf_pass': r.obf_pass,
                    'opt_level': r.opt_level,
                    'compiles': r.compiles,
                    'passes_tests': r.passes_tests,
                    'levenshtein': r.levenshtein,
                    'codebleu': r.codebleu,
                }
                for r in results_original
            ]
        }, f, indent=2)

    with open(output_dir / "bad_codebleu_original.json", 'w') as f:
        json.dump(bad_codebleu_original, f, indent=2)
    if original_cache_path.exists():
        original_cache_path.unlink()

    del model_original
    torch.cuda.empty_cache()

    model_finetuned = DecompileModel(
        config.finetuned_model,
        "Fine-tuned LLM4Decompile-9B",
        max_sequence_len=config.max_sequence_len,
    )

    results_finetuned, bad_codebleu_finetuned = evaluate_model(model_finetuned, samples, config, finetuned_cache_path)
    stats_finetuned = compute_statistics(results_finetuned, "Fine-tuned LLM4Decompile-9B")
    print_statistics(stats_finetuned)

    with open(output_dir / "results_finetuned.json", 'w') as f:
        json.dump({
            'stats': stats_finetuned,
            'results': [
                {
                    'idx': r.idx,
                    'obf_pass': r.obf_pass,
                    'opt_level': r.opt_level,
                    'compiles': r.compiles,
                    'passes_tests': r.passes_tests,
                    'levenshtein': r.levenshtein,
                    'codebleu': r.codebleu,
                }
                for r in results_finetuned
            ]
        }, f, indent=2)

    with open(output_dir / "bad_codebleu_finetuned.json", 'w') as f:
        json.dump(bad_codebleu_finetuned, f, indent=2)
    if finetuned_cache_path.exists():
        finetuned_cache_path.unlink()

    print_comparison(stats_original, stats_finetuned)

    summary = {
        'config': {
            'original_model': config.original_model,
            'finetuned_model': config.finetuned_model,
            'num_samples': len(samples),
        },
        'original': stats_original,
        'finetuned': stats_finetuned,
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
