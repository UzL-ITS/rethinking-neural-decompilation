#!/usr/bin/env python3
"""
Compute only the execution-guided cache for the shared reranking pipeline.

This script is intended for the case where:
- `candidates_<num_samples>_<num_candidates>.pkl` already exists
- other metric caches may already exist
- you only need `phase3_5_execution_guided.pkl` without rerunning the full
  `evaluate_all_metrics.py` pipeline
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import multiprocessing as mp
import pickle
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


@dataclass
class Config:
    num_samples: int = 656
    num_candidates: int = 100
    cache_dir: str = "./cache"
    num_workers: int = 20
    max_guided_checks: int = 3


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


def build_combined_program(code: str, c_func: str, c_test: str) -> str:
    """Combine includes, generated code, and test harness into one C file."""
    c_include = ""
    seen_includes = set()

    for source in (c_func, c_test):
        for line in source.split("\n"):
            if "#include" in line and line not in seen_includes:
                c_include += line + "\n"
                seen_includes.add(line)

    c_test_clean = "\n".join(line for line in c_test.split("\n") if "#include" not in line)
    return c_include + "\n" + code + "\n" + c_test_clean


def compile_c_file(c_file: str, exe_base: str, timeout: int = 10) -> Tuple[bool, str]:
    """Compile C source into an executable and return (success, stderr)."""
    compile_cmd = ["gcc", c_file, "-o", exe_base, "-lm", "-w"]
    try:
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stderr.decode(errors="replace")[:500]
    except subprocess.TimeoutExpired:
        return False, "Compilation timeout"
    except Exception as exc:
        return False, str(exc)


def run_binary(exe_base: str, timeout: int = 5) -> bool:
    """Run compiled binary with a platform-appropriate command."""
    exe_path = exe_base + (".exe" if os.name == "nt" else "")

    if os.name != "nt":
        sandbox_script = Path(__file__).resolve().parent / "run-sandboxed.sh"
        if sandbox_script.exists():
            try:
                result = subprocess.run(
                    ["bash", str(sandbox_script), exe_path, f"{timeout}s"],
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


def check_compilation(args: Tuple[str, str, str]) -> Tuple[bool, str]:
    """Check if code compiles. Returns (compiles, error_message)."""
    code, c_func, c_test = args
    c_combine = build_combined_program(code, c_func, c_test)

    with tempfile.TemporaryDirectory() as temp_dir:
        c_file = os.path.join(temp_dir, "test.c")
        exe = os.path.join(temp_dir, "test")

        with open(c_file, "w", encoding="utf-8") as handle:
            handle.write(c_combine)

        return compile_c_file(c_file, exe, timeout=10)


def run_test(args: Tuple[str, str, str]) -> bool:
    """Run compiled code and check if tests pass."""
    code, c_func, c_test = args
    c_combine = build_combined_program(code, c_func, c_test)

    with tempfile.TemporaryDirectory() as temp_dir:
        c_file = os.path.join(temp_dir, "test.c")
        exe = os.path.join(temp_dir, "test")

        with open(c_file, "w", encoding="utf-8") as handle:
            handle.write(c_combine)

        try:
            compiled, _ = compile_c_file(c_file, exe, timeout=10)
            if not compiled:
                return False
            return run_binary(exe, timeout=5)
        except Exception:
            return False


def derive_execution_checkpoints(c_test: str, max_checks: int) -> List[str]:
    """
    Build partial test harnesses by keeping prefixes of assert lines.

    The full suite is excluded so the score uses partial execution evidence
    rather than the final evaluation label.
    """
    lines = c_test.split("\n")
    assert_indices = [i for i, line in enumerate(lines) if "assert(" in line]

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
            if "assert(" not in line:
                partial_lines.append(line)
                continue
            if kept < keep_count:
                partial_lines.append(line)
            kept += 1
        checkpoints.append("\n".join(partial_lines))

    return checkpoints


def compute_execution_guided_score(code: str, c_func: str, c_test: str, max_checks: int) -> float:
    """Score a candidate by how many partial execution checkpoints it passes."""
    c_test_clean = "\n".join(line for line in c_test.split("\n") if "#include" not in line)
    checkpoints = derive_execution_checkpoints(c_test_clean, max_checks)

    if not checkpoints:
        return 0.0

    passed = 0
    for checkpoint in checkpoints:
        if run_test((code, c_func, checkpoint)):
            passed += 1

    return passed / len(checkpoints)


def compute_execution_guided_score_from_args(args: Tuple[str, str, str, int]) -> float:
    return compute_execution_guided_score(*args)


def load_samples_from_cache(config: Config) -> List[Sample]:
    cache_file = Path(config.cache_dir) / f"candidates_{config.num_samples}_{config.num_candidates}.pkl"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Missing candidate cache: {cache_file}. This script expects the shared "
            "candidate cache to already exist."
        )

    print(f"Loading cached candidates from {cache_file}")
    with open(cache_file, "rb") as handle:
        samples = pickle.load(handle)

    print(f"Loaded {len(samples)} samples")
    return samples


def load_or_compute_compilation(samples: List[Sample], config: Config) -> None:
    """Reuse phase1 compilation cache when available; otherwise compute it once."""
    compile_cache = Path(config.cache_dir) / "phase1_compilation.pkl"

    if compile_cache.exists():
        print("Loading cached compilation results")
        with open(compile_cache, "rb") as handle:
            compile_results = pickle.load(handle)

        for sample in samples:
            for i, candidate in enumerate(sample.candidates):
                candidate.compiles = compile_results.get((sample.idx, i), False)
        return

    print("Compilation cache missing; computing phase1_compilation.pkl")
    tasks = []
    task_info = []

    for sample in samples:
        for i, candidate in enumerate(sample.candidates):
            tasks.append((candidate.code, sample.c_func, sample.c_test))
            task_info.append((sample.idx, i))

    with mp.Pool(config.num_workers) as pool:
        results = list(tqdm(pool.imap(check_compilation, tasks), total=len(tasks), desc="Compiling"))

    compile_results = {}
    sample_dict = {sample.idx: sample for sample in samples}
    for (sample_idx, cand_idx), (compiles, _) in zip(task_info, results):
        sample_dict[sample_idx].candidates[cand_idx].compiles = compiles
        compile_results[(sample_idx, cand_idx)] = compiles

    with open(compile_cache, "wb") as handle:
        pickle.dump(compile_results, handle)
    print(f"Saved compilation cache to {compile_cache}")


def compute_execution_guided_cache(samples: List[Sample], config: Config) -> Dict[Tuple[int, int], Dict[str, float]]:
    execution_cache = Path(config.cache_dir) / "phase3_5_execution_guided.pkl"

    if execution_cache.exists():
        print(f"Execution-guided cache already exists at {execution_cache}; overwriting it")

    execution_results = {}
    for sample in tqdm(samples, desc="Execution-guided"):
        compiling = [(i, c) for i, c in enumerate(sample.candidates) if c.compiles]
        if not compiling:
            continue

        candidate_indices = [i for i, _ in compiling]
        candidate_objs = [c for _, c in compiling]
        tasks = [
            (candidate.code, sample.c_func, sample.c_test, config.max_guided_checks)
            for candidate in candidate_objs
        ]

        with ThreadPoolExecutor(max_workers=min(len(tasks), config.num_workers)) as executor:
            scores = list(executor.map(compute_execution_guided_score_from_args, tasks))

        for cand_idx, score in zip(candidate_indices, scores):
            execution_results[(sample.idx, cand_idx)] = {"execution_guided": score}

    with open(execution_cache, "wb") as handle:
        pickle.dump(execution_results, handle)

    print(f"Saved execution-guided cache to {execution_cache}")
    return execution_results


def main() -> None:
    config = Config()
    Path(config.cache_dir).mkdir(exist_ok=True)

    print("=" * 70)
    print("EXECUTION-GUIDED CACHE BUILDER")
    print("=" * 70)
    print(f"Samples: {config.num_samples}")
    print(f"Candidates per sample: {config.num_candidates}")
    print(f"Workers: {config.num_workers}")
    print(f"Guided checkpoints: {config.max_guided_checks}")

    samples = load_samples_from_cache(config)
    load_or_compute_compilation(samples, config)
    compute_execution_guided_cache(samples, config)

    print("=" * 70)


if __name__ == "__main__":
    main()
