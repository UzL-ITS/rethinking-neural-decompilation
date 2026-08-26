#!/usr/bin/env python3
"""
Compute oracle@k from cached RQ2 candidate/test results.

This script does not rerun generation, compilation, or testing.
It reconstructs candidate state from cache files and computes oracle pass rate
for every pool size k = 1..N, where N is the maximum number of compiling
candidates available for any sample.

By default, the denominator matches the existing RQ2 evaluation logic:
- only samples with at least one compiling candidate are counted

Outputs:
- oracle_curve.json
- oracle_curve.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


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
    real_io_pairs: dict | None = None
    real_iospec: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute oracle@k curve from RQ2 cache files.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./cache"),
        help="Directory containing candidates_*.pkl, phase1_compilation.pkl, phase4_test_results.pkl",
    )
    parser.add_argument(
        "--candidates-file",
        type=Path,
        default=None,
        help="Optional explicit path to candidates_*.pkl. If omitted, the script auto-selects the only match.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path. Default: <cache-dir>/oracle_curve.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Default: <cache-dir>/oracle_curve.csv",
    )
    return parser.parse_args()


def resolve_candidates_file(cache_dir: Path, explicit_file: Path | None) -> Path:
    if explicit_file is not None:
        if not explicit_file.exists():
            raise FileNotFoundError(f"Candidates file not found: {explicit_file}")
        return explicit_file

    matches = sorted(cache_dir.glob("candidates_*.pkl"))
    if not matches:
        raise FileNotFoundError(f"No candidates_*.pkl found in {cache_dir}")
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise RuntimeError(
            f"Multiple candidate files found in {cache_dir}. "
            f"Pass --candidates-file explicitly. Found: {names}"
        )
    return matches[0]


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def attach_cache_state(samples: List[Sample], compile_results: Dict, test_results: Dict) -> None:
    for sample in samples:
        for cand_idx, candidate in enumerate(sample.candidates):
            key = (sample.idx, cand_idx)
            if key in compile_results:
                candidate.compiles = bool(compile_results[key])
            if key in test_results:
                candidate.passes_tests = bool(test_results[key])


def compute_oracle_curve(samples: List[Sample]) -> Dict:
    compiling_prefix_passes: List[List[bool]] = []
    samples_with_candidates = 0
    samples_with_compiling = 0

    for sample in samples:
        if sample.candidates:
            samples_with_candidates += 1

        compiling = [c for c in sample.candidates if c.compiles]
        if not compiling:
            continue

        samples_with_compiling += 1
        compiling_prefix_passes.append([bool(c.passes_tests) for c in compiling])

    if not compiling_prefix_passes:
        raise RuntimeError("No samples with compiling candidates found. Cannot compute oracle curve.")

    max_compiling_candidates = max(len(passes) for passes in compiling_prefix_passes)
    points = []

    for k in range(1, max_compiling_candidates + 1):
        correct = 0
        for passes in compiling_prefix_passes:
            if any(passes[:k]):
                correct += 1

        total = samples_with_compiling
        pass_rate = (correct / total * 100.0) if total else 0.0
        points.append(
            {
                "k": k,
                "correct": correct,
                "total": total,
                "pass_rate": pass_rate,
            }
        )

    first_plateau_k = max_compiling_candidates
    final_correct = points[-1]["correct"]
    for point in points:
        if point["correct"] == final_correct:
            first_plateau_k = point["k"]
            break

    return {
        "summary": {
            "samples_total": len(samples),
            "samples_with_any_candidates": samples_with_candidates,
            "samples_with_compiling_candidates": samples_with_compiling,
            "max_compiling_candidates_per_sample": max_compiling_candidates,
            "oracle_final_correct": final_correct,
            "oracle_final_total": points[-1]["total"],
            "oracle_final_pass_rate": points[-1]["pass_rate"],
            "first_plateau_k": first_plateau_k,
        },
        "oracle_curve": points,
    }


def save_outputs(results: Dict, output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "correct", "total", "pass_rate"])
        writer.writeheader()
        writer.writerows(results["oracle_curve"])


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    candidates_file = resolve_candidates_file(cache_dir, args.candidates_file.resolve() if args.candidates_file else None)
    compile_file = cache_dir / "phase1_compilation.pkl"
    test_file = cache_dir / "phase4_test_results.pkl"

    if not compile_file.exists():
        raise FileNotFoundError(f"Compilation cache not found: {compile_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test-results cache not found: {test_file}")

    output_json = (args.output_json.resolve() if args.output_json else cache_dir / "oracle_curve.json")
    output_csv = (args.output_csv.resolve() if args.output_csv else cache_dir / "oracle_curve.csv")

    print(f"Loading candidates from {candidates_file}")
    samples = load_pickle(candidates_file)
    print(f"Loaded {len(samples)} samples")

    print(f"Loading compilation cache from {compile_file}")
    compile_results = load_pickle(compile_file)
    print(f"Compilation cache entries: {len(compile_results):,}")

    print(f"Loading test-results cache from {test_file}")
    test_results = load_pickle(test_file)
    print(f"Test-results cache entries: {len(test_results):,}")

    attach_cache_state(samples, compile_results, test_results)
    results = compute_oracle_curve(samples)

    summary = results["summary"]
    print("\nSummary:")
    print(f"  Samples total:                   {summary['samples_total']:,}")
    print(f"  Samples with any candidates:     {summary['samples_with_any_candidates']:,}")
    print(f"  Samples with compiling candidates:{summary['samples_with_compiling_candidates']:,}")
    print(f"  Max compiling candidates/sample: {summary['max_compiling_candidates_per_sample']:,}")
    print(f"  Final oracle pass rate:          {summary['oracle_final_pass_rate']:.1f}%")
    print(f"  First plateau k:                 {summary['first_plateau_k']:,}")

    save_outputs(results, output_json, output_csv)
    print(f"\nSaved JSON to {output_json}")
    print(f"Saved CSV to {output_csv}")


if __name__ == "__main__":
    main()
