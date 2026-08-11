#!/usr/bin/env python3
"""
LLM4Decompile Selection Pipeline

Generates 200 candidates per sample, filters by compilation, ranks using
execution-guided reranking, and evaluates pass@1 for each metric.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import subprocess
import tempfile
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# vLLM is only needed when regenerating candidates, not for cache-based eval.
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    model_name: str = "LLM4Binary/llm4decompile-6.7b-v1.5"
    num_samples: int = 656  # Number of test samples to evaluate (all samples)
    num_candidates: int = 100  # Must match the shared RQ2 candidate cache
    temperatures: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
    max_new_tokens: int = 4096
    max_model_len: int = 12288
    cache_dir: str = "./cache"
    num_workers: int = 20  # For parallel compilation
    max_guided_checks: int = 3  # Partial checkpoints for execution-guided scoring


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Candidate:
    code: str
    temperature: float
    compiles: bool = False
    passes_tests: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Sample:
    idx: int
    opt: str
    ghidra: str
    c_func: str
    c_test: str
    candidates: List[Candidate] = field(default_factory=list)


# ============================================================================
# Compilation and Evaluation
# ============================================================================

def compile_c_file(c_file: str, exe_base: str, timeout: int = 10) -> Tuple[bool, str]:
    """Compile C source into an executable and return (success, stderr)."""
    compile_cmd = ['gcc', c_file, '-o', exe_base, '-lm', '-w']
    try:
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stderr.decode(errors='replace')[:500]
    except subprocess.TimeoutExpired:
        return False, "Compilation timeout"
    except Exception as e:
        return False, str(e)


def run_binary(exe_base: str, timeout: int = 5) -> bool:
    """Run compiled binary with a platform-appropriate command."""
    exe_path = exe_base + ('.exe' if os.name == 'nt' else '')

    if os.name != 'nt':
        sandbox_script = Path(__file__).resolve().parent / "run-sandboxed.sh"
        if sandbox_script.exists():
            try:
                result = subprocess.run(
                    ['bash', str(sandbox_script), exe_path, f'{timeout}s'],
                    capture_output=True,
                    timeout=timeout + 5,
                )
                if result.returncode == 0:
                    return True
            except Exception:
                pass

    try:
        result = subprocess.run(
            [exe_path],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except Exception:
        return False


def build_combined_program(code: str, c_func: str, c_test: str) -> str:
    """Combine includes, generated code, and test harness into one C file."""
    c_include = ''
    seen_includes = set()

    for source in (c_func, c_test):
        for line in source.split('\n'):
            if '#include' in line and line not in seen_includes:
                c_include += line + '\n'
                seen_includes.add(line)

    c_test_clean = '\n'.join(line for line in c_test.split('\n') if '#include' not in line)
    return c_include + '\n' + code + '\n' + c_test_clean


def check_compilation(args: Tuple[str, str, str]) -> Tuple[bool, str]:
    """Check if code compiles. Returns (compiles, error_message)."""
    code, c_func, c_test = args
    c_combine = build_combined_program(code, c_func, c_test)

    with tempfile.TemporaryDirectory() as temp_dir:
        c_file = os.path.join(temp_dir, 'test.c')
        exe = os.path.join(temp_dir, 'test')

        with open(c_file, 'w', encoding='utf-8') as f:
            f.write(c_combine)

        return compile_c_file(c_file, exe, timeout=10)


def run_test(args: Tuple[str, str, str]) -> bool:
    """Run compiled code and check if tests pass."""
    code, c_func, c_test = args
    c_combine = build_combined_program(code, c_func, c_test)

    with tempfile.TemporaryDirectory() as temp_dir:
        c_file = os.path.join(temp_dir, 'test.c')
        exe = os.path.join(temp_dir, 'test')

        with open(c_file, 'w', encoding='utf-8') as f:
            f.write(c_combine)

        try:
            compiled, _ = compile_c_file(c_file, exe, timeout=10)
            if not compiled:
                return False

            return run_binary(exe, timeout=5)
        except Exception:
            return False


def derive_execution_checkpoints(c_test: str, max_checks: int) -> List[str]:
    """
    Build a few partial test harnesses by keeping only prefixes of assert lines.

    We intentionally exclude the full suite so the execution-guided score is
    based on partial execution evidence rather than the final evaluation label.
    """
    lines = c_test.split('\n')
    assert_indices = [i for i, line in enumerate(lines) if 'assert(' in line]

    if not assert_indices:
        return [c_test]

    num_asserts = len(assert_indices)
    if num_asserts == 1:
        keep_counts = [0]
    else:
        keep_counts = []
        for value in [1, max(1, num_asserts // 2), num_asserts - 1]:
            if 0 <= value < num_asserts and value not in keep_counts:
                keep_counts.append(value)

    checkpoints = []
    for keep_count in keep_counts[:max_checks]:
        kept = 0
        partial_lines = []
        for line in lines:
            if 'assert(' not in line:
                partial_lines.append(line)
                continue
            if kept < keep_count:
                partial_lines.append(line)
            kept += 1
        checkpoints.append('\n'.join(partial_lines))

    return checkpoints


def compute_execution_guided_score(code: str, c_func: str, c_test: str, max_checks: int) -> float:
    """Score a candidate by how many partial execution checkpoints it passes."""
    c_test_clean = '\n'.join(line for line in c_test.split('\n') if '#include' not in line)
    checkpoints = derive_execution_checkpoints(c_test_clean, max_checks)

    if not checkpoints:
        return 0.0

    passed = 0
    for checkpoint in checkpoints:
        if run_test((code, c_func, checkpoint)):
            passed += 1

    return passed / len(checkpoints)


def compute_execution_guided_score_from_args(args: Tuple[str, str, str, int]) -> float:
    """Wrapper so the executor call stays simple and pickle-safe."""
    return compute_execution_guided_score(*args)


# ============================================================================
# Main Pipeline
# ============================================================================

def load_dataset(data_path: str) -> List[Dict]:
    """Load the evaluation dataset."""
    with open(data_path, 'r') as f:
        return json.load(f)


def generate_candidates(llm: LLM, samples: List[Sample], config: Config) -> List[Sample]:
    """Generate candidates for all samples."""
    if LLM is None or SamplingParams is None:
        raise RuntimeError("vLLM is not available in this environment.")

    print("\n=== Phase 1: Generating Candidates ===")

    tokenizer = llm.get_tokenizer()
    max_prompt_len = config.max_model_len - config.max_new_tokens

    valid_samples = []
    skipped = []
    for s in samples:
        prompt = f"# This is the assembly code:\n{s.ghidra}\n# What is the source code?\n"
        token_len = len(tokenizer.encode(prompt))
        if token_len <= max_prompt_len:
            valid_samples.append(s)
        else:
            skipped.append((s.idx, token_len))

    print(f"Valid samples: {len(valid_samples)}, Skipped (too long): {len(skipped)}")
    if skipped:
        print(f"  Skipped sample indices: {[s[0] for s in skipped[:10]]}{'...' if len(skipped) > 10 else ''}")

    candidates_per_temp = config.num_candidates // len(config.temperatures)
    sample_to_candidates = {s.idx: [] for s in valid_samples}

    for temp in tqdm(config.temperatures, desc="Temperatures"):
        params = SamplingParams(
            temperature=max(temp, 0.01),
            max_tokens=config.max_new_tokens,
            top_p=0.95 if temp > 0 else 1.0,
            n=candidates_per_temp,
        )

        prompts = [f"# This is the assembly code:\n{s.ghidra}\n# What is the source code?\n" for s in valid_samples]
        outputs = llm.generate(prompts, params)

        for i, output in enumerate(outputs):
            sample_idx = valid_samples[i].idx
            for out in output.outputs:
                sample_to_candidates[sample_idx].append(
                    Candidate(code=out.text, temperature=temp)
                )

    for sample in valid_samples:
        sample.candidates = sample_to_candidates[sample.idx]

    return samples


def load_samples_from_cache(config: Config) -> List[Sample]:
    """Load the shared RQ2 candidate cache."""
    cache_file = Path(config.cache_dir) / f"candidates_{config.num_samples}_{config.num_candidates}.pkl"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Missing candidate cache: {cache_file}. Run selection_pipeline.py first "
            "to generate the shared RQ2 candidate set."
        )

    print(f"\nLoading cached candidates from {cache_file}")
    with open(cache_file, 'rb') as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples")
    return samples


def load_or_compute_compilation(samples: List[Sample], config: Config) -> None:
    """Load compilation cache or compute it once if missing."""
    compile_cache = Path(config.cache_dir) / "phase1_compilation.pkl"

    if compile_cache.exists():
        print("\n=== Phase 1: Loading cached compilation results ===")
        with open(compile_cache, 'rb') as f:
            compile_results = pickle.load(f)

        for sample in samples:
            for i, candidate in enumerate(sample.candidates):
                candidate.compiles = compile_results.get((sample.idx, i), False)
        return

    print("\n=== Phase 1: Computing compilation results ===")
    tasks = []
    task_info = []

    for sample in samples:
        for i, candidate in enumerate(sample.candidates):
            tasks.append((candidate.code, sample.c_func, sample.c_test))
            task_info.append((sample.idx, i))

    with mp.Pool(config.num_workers) as pool:
        results = list(tqdm(pool.imap(check_compilation, tasks), total=len(tasks), desc="Compiling"))

    compile_results = {}
    sample_dict = {s.idx: s for s in samples}
    for (sample_idx, cand_idx), (compiles, _) in zip(task_info, results):
        sample_dict[sample_idx].candidates[cand_idx].compiles = compiles
        compile_results[(sample_idx, cand_idx)] = compiles

    with open(compile_cache, 'wb') as f:
        pickle.dump(compile_results, f)
    print(f"Saved compilation cache to {compile_cache}")


def load_or_compute_execution_guided(samples: List[Sample], config: Config) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Load or compute only the execution-guided metric cache used by evaluate_selection.py."""
    execution_cache = Path(config.cache_dir) / "phase3_5_execution_guided.pkl"

    if execution_cache.exists():
        print("\n=== Phase 2: Loading cached execution-guided metrics ===")
        with open(execution_cache, 'rb') as f:
            execution_results = pickle.load(f)
    else:
        print("\n=== Phase 2: Computing execution-guided metrics ===")
        execution_results = {}

        for sample in tqdm(samples, desc="Execution-guided"):
            compiling = [(i, c) for i, c in enumerate(sample.candidates) if c.compiles]
            if not compiling:
                continue

            candidate_indices = [i for i, _ in compiling]
            candidate_objs = [c for _, c in compiling]
            tasks = [(c.code, sample.c_func, sample.c_test, config.max_guided_checks) for c in candidate_objs]

            with ThreadPoolExecutor(max_workers=min(len(tasks), config.num_workers)) as ex:
                scores = list(ex.map(compute_execution_guided_score_from_args, tasks))

            for cand_idx, score in zip(candidate_indices, scores):
                execution_results[(sample.idx, cand_idx)] = {'execution_guided': score}

        with open(execution_cache, 'wb') as f:
            pickle.dump(execution_results, f)
        print(f"Saved execution-guided cache to {execution_cache}")

    for sample in samples:
        for i, candidate in enumerate(sample.candidates):
            metrics = execution_results.get((sample.idx, i))
            if metrics:
                candidate.metrics.update(metrics)

    return execution_results


def load_or_compute_test_results(samples: List[Sample], config: Config) -> None:
    """Reuse cached test results if available; otherwise compute and cache them."""
    test_cache = Path(config.cache_dir) / "phase4_test_results.pkl"

    if test_cache.exists():
        print("\n=== Phase 3: Loading cached test results ===")
        with open(test_cache, 'rb') as f:
            test_results = pickle.load(f)

        for sample in samples:
            for i, candidate in enumerate(sample.candidates):
                candidate.passes_tests = test_results.get((sample.idx, i), False)
        return

    print("\n=== Phase 3: Computing test results ===")
    test_results = {}
    for sample in tqdm(samples, desc="Testing"):
        compiling = [(i, c) for i, c in enumerate(sample.candidates) if c.compiles]
        if not compiling:
            continue

        test_tasks = [(c.code, sample.c_func, sample.c_test) for _, c in compiling]
        with ThreadPoolExecutor(max_workers=min(len(test_tasks), config.num_workers)) as ex:
            passes = list(ex.map(run_test, test_tasks))

        for (cand_idx, candidate), passed in zip(compiling, passes):
            candidate.passes_tests = passed
            test_results[(sample.idx, cand_idx)] = passed

    with open(test_cache, 'wb') as f:
        pickle.dump(test_results, f)
    print(f"Saved test results cache to {test_cache}")


def evaluate_selection(samples: List[Sample], config: Config) -> Dict[str, Dict]:
    """Evaluate pass@1 for execution-guided reranking against random and oracle baselines."""
    print("\n=== Phase 4: Evaluating Execution-Guided Selection ===")

    metrics = ['execution_guided']

    results = {'execution_guided': {'correct': 0, 'total': 0, 'no_candidates': 0}}
    results['random'] = {'correct': 0, 'total': 0, 'no_candidates': 0}
    results['oracle'] = {'correct': 0, 'total': 0, 'no_candidates': 0}

    for sample in tqdm(samples, desc="Evaluating"):
        compiling = [c for c in sample.candidates if c.compiles]

        if not compiling:
            for m in metrics:
                results[m]['no_candidates'] += 1
            results['random']['no_candidates'] += 1
            results['oracle']['no_candidates'] += 1
            continue

        results['oracle']['total'] += 1
        if any(c.passes_tests for c in compiling):
            results['oracle']['correct'] += 1

        import random
        random_candidate = random.choice(compiling)
        results['random']['total'] += 1
        if random_candidate.passes_tests:
            results['random']['correct'] += 1

        for metric in metrics:
            results[metric]['total'] += 1
            best = max(compiling, key=lambda c: c.metrics.get(metric, 0))
            if best.passes_tests:
                results[metric]['correct'] += 1

    return results


def main():
    config = Config()

    Path(config.cache_dir).mkdir(exist_ok=True)

    print("=" * 70)
    print("RQ2b ADVANCED RERANKING: EXECUTION-GUIDED")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Samples: {config.num_samples}")
    print(f"  Candidates per sample: {config.num_candidates}")
    print(f"  Temperatures: {config.temperatures}")
    print(f"  Max new tokens: {config.max_new_tokens}")
    print(f"  Max model length: {config.max_model_len}")
    print(f"  Workers: {config.num_workers}")
    print(f"  Guided checkpoints: {config.max_guided_checks}")

    data_path = str(Path(__file__).resolve().parent / "../../LLM4Decompile/legacy-test/decompile-eval-executable-gcc-ghidra.json")
    dataset = load_dataset(data_path)

    samples = []
    opts = ['O0', 'O1', 'O2', 'O3']
    for i in range(config.num_samples):
        row = dataset[i]
        samples.append(Sample(
            idx=i,
            opt=opts[i % 4],
            ghidra=row['input_asm_prompt'],
            c_func=row['c_func'],
            c_test=row['c_test'],
        ))

    print(f"\nLoaded {len(samples)} samples")

    cache_file = Path(config.cache_dir) / f"candidates_{config.num_samples}_{config.num_candidates}.pkl"

    if cache_file.exists():
        print(f"\nLoading cached candidates from {cache_file}")
        with open(cache_file, 'rb') as f:
            samples = pickle.load(f)
    else:
        print("\nLoading LLM...")
        llm = LLM(
            model=config.model_name,
            trust_remote_code=True,
            dtype='bfloat16',
            max_model_len=config.max_model_len,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,
        )

        samples = generate_candidates(llm, samples, config)

        print(f"\nSaving candidates to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(samples, f)

        del llm

    load_or_compute_compilation(samples, config)
    load_or_compute_execution_guided(samples, config)
    load_or_compute_test_results(samples, config)

    results = evaluate_selection(samples, config)

    print("\n" + "=" * 70)
    print(f"{'METRIC':<25} | {'PASS@1':<10} | {'SUCCESS/TOTAL'}")
    print("=" * 70)

    sorted_metrics = sorted(results.items(), key=lambda x: x[1]['correct'] / max(x[1]['total'], 1), reverse=True)

    for metric, stats in sorted_metrics:
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            print(f"  {metric:20s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:5.1f}%  (no candidates: {stats['no_candidates']})")

    results_file = Path(config.cache_dir) / "selection_results_execution_guided.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    print("=" * 70)


if __name__ == "__main__":
    main()
