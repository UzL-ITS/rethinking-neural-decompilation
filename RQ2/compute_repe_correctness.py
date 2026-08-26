#!/usr/bin/env python3
"""
Compute a RepE/LAT-style correctness reranking cache for the HumanEval C harness.

Inputs:
- candidates_{num_samples}_{num_candidates}.pkl
- phase1_compilation.pkl
- phase4_test_results.pkl

Output:
- phase3_7_repe_correctness.pkl
- phase3_7_repe_model_humaneval.pt
- repe_correctness_results_test_humaneval.json

This script does not compile or execute code. It uses the cached compilation
and test labels from evaluate_all_metrics.py, fits a correctness direction from
hidden-state differences between passing and failing candidates, then scores
all compiling candidates with a new metric: repe_correctness.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class TrainConfig:
    """Present only so old checkpoint/candidate pickles can unpickle safely."""

    data_path: str = "unused-but-required-for-unpickling"
    test_samples: int = 100
    max_length: int = 512
    model_name: str = "microsoft/unixcoder-base"
    projection_dim: int = 256
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    temperature: float = 0.07
    weight_decay: float = 0.01
    output_dir: str = "./checkpoints"
    save_every: int = 2


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RepE correctness scores for HumanEval candidate reranking."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("./cache"))
    parser.add_argument("--candidates-file", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--model-name", type=str, default="LLM4Binary/llm4decompile-9b-v2")
    parser.add_argument("--fit-samples", type=int, default=100)
    parser.add_argument("--val-samples", type=int, default=100)
    parser.add_argument("--max-pairs-per-split", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite-scores", action="store_true")
    return parser.parse_args()


def infer_seed_from_name(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not infer seed from filename: {path.name}")
    return int(match.group(1))


def infer_num_samples_from_name(path: Path) -> int:
    match = re.search(r"candidates_(\d+)_\d+_seed\d+\.pkl$", path.name)
    if not match:
        raise ValueError(f"Could not infer num_samples from filename: {path.name}")
    return int(match.group(1))


def resolve_input_paths(cache_dir: Path, candidates_file: Path | None, seed: int | None, num_samples: int | None, num_candidates: int):
    if candidates_file is not None:
        candidates_path = candidates_file
    else:
        if seed is not None and num_samples is not None:
            candidates_path = cache_dir / f"candidates_{num_samples}_{num_candidates}_seed{seed}.pkl"
        elif seed is not None:
            matches = sorted(cache_dir.glob(f"candidates_*_{num_candidates}_seed{seed}.pkl"))
            if not matches:
                raise FileNotFoundError(f"No candidate pickle found for seed {seed} in {cache_dir}")
            candidates_path = matches[0]
        else:
            matches = sorted(cache_dir.glob(f"candidates_*_{num_candidates}_seed*.pkl"))
            if len(matches) == 1:
                candidates_path = matches[0]
            elif not matches:
                raise FileNotFoundError(f"No candidate pickles found in {cache_dir}")
            else:
                raise RuntimeError(
                    f"Multiple candidate pickles found in {cache_dir}; pass --seed or --candidates-file explicitly."
                )

    resolved_seed = seed if seed is not None else infer_seed_from_name(candidates_path)
    resolved_num_samples = num_samples if num_samples is not None else infer_num_samples_from_name(candidates_path)

    compilation_file = cache_dir / f"phase1_compilation_seed{resolved_seed}.pkl"
    test_file = cache_dir / f"phase4_test_results_seed{resolved_seed}.pkl"
    output_cache = cache_dir / f"phase3_7_repe_correctness_seed{resolved_seed}.pkl"
    model_cache = cache_dir / f"phase3_7_repe_model_humaneval_seed{resolved_seed}.pt"
    results_file = cache_dir / f"repe_correctness_results_test_humaneval_seed{resolved_seed}.json"

    return {
        "seed": resolved_seed,
        "num_samples": resolved_num_samples,
        "candidates_file": candidates_path,
        "compilation_file": compilation_file,
        "test_file": test_file,
        "output_cache": output_cache,
        "model_cache": model_cache,
        "results_file": results_file,
    }


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def apply_cached_labels(samples: List[Sample], compile_results: dict, test_results: dict) -> None:
    """Attach cached labels. Missing compilation entries are treated as non-compiling."""
    for sample in samples:
        for i, candidate in enumerate(sample.candidates):
            key = (sample.idx, i)
            candidate.compiles = bool(compile_results.get(key, False))
            candidate.passes_tests = bool(test_results.get(key, False)) if candidate.compiles else False


def split_samples(samples: List[Sample], fit_count: int, val_count: int) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    fit = samples[:fit_count]
    val = samples[fit_count:fit_count + val_count]
    test = samples[fit_count + val_count:]
    return fit, val, test


def build_pairs(samples: List[Sample], max_pairs: int, rng: random.Random) -> List[Tuple[Sample, Candidate, Candidate]]:
    """Build correct/incorrect candidate pairs from the same HumanEval task."""
    pairs: List[Tuple[Sample, Candidate, Candidate]] = []
    for sample in samples:
        passing = [c for c in sample.candidates if c.compiles and c.passes_tests]
        failing = [c for c in sample.candidates if c.compiles and not c.passes_tests]
        if not passing or not failing:
            continue

        rng.shuffle(passing)
        rng.shuffle(failing)
        for good, bad in zip(passing, failing):
            pairs.append((sample, good, bad))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def make_prompt(sample: Sample, code: str) -> str:
    """Prompt mirrors the paper but uses Ghidra output as the task context."""
    return (
        "Consider the amount of correctness in the following code.\n"
        "Task:\n"
        f"{sample.ghidra}\n\n"
        "Code:\n"
        "```c\n"
        f"{code}\n"
        "```\n\n"
        "The amount of correctness in the code is"
    )


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
    model.eval()
    return tokenizer, model


def extract_hidden_states(
    prompts: List[str],
    tokenizer,
    model,
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    """Return tensor with shape [num_layers, num_prompts, hidden_size]."""
    collected: List[torch.Tensor] = []
    input_device = next(model.parameters()).device

    for start in tqdm(range(0, len(prompts), batch_size), desc="Hidden states"):
        batch = prompts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        last_positions = encoded["attention_mask"].sum(dim=1) - 1

        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)

        layer_values = []
        for hidden in outputs.hidden_states[1:]:
            hidden_positions = last_positions.to(hidden.device)
            rows = hidden[torch.arange(hidden.shape[0], device=hidden.device), hidden_positions]
            layer_values.append(rows.detach().float().cpu())

        collected.append(torch.stack(layer_values, dim=0))

        del outputs, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(collected, dim=1)


def fit_directions(correct_hidden: torch.Tensor, wrong_hidden: torch.Tensor) -> torch.Tensor:
    """Fit one PCA direction per layer. Shape: [num_layers, hidden_size]."""
    diffs = correct_hidden - wrong_hidden
    directions = []

    for layer_idx in tqdm(range(diffs.shape[0]), desc="Fitting PCA directions"):
        layer_diffs = diffs[layer_idx]
        centered = layer_diffs - layer_diffs.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        direction = F.normalize(vh[0], dim=0)

        correct_scores = correct_hidden[layer_idx] @ direction
        wrong_scores = wrong_hidden[layer_idx] @ direction
        if correct_scores.mean() < wrong_scores.mean():
            direction = -direction
        directions.append(direction)

    return torch.stack(directions, dim=0)


def choose_best_layer(
    directions: torch.Tensor,
    correct_hidden: torch.Tensor,
    wrong_hidden: torch.Tensor,
) -> Tuple[int, List[float]]:
    accuracies = []
    for layer_idx in range(directions.shape[0]):
        direction = directions[layer_idx]
        correct_scores = correct_hidden[layer_idx] @ direction
        wrong_scores = wrong_hidden[layer_idx] @ direction
        accuracies.append(float((correct_scores > wrong_scores).float().mean().item()))

    best_layer = max(range(len(accuracies)), key=lambda idx: accuracies[idx])
    return best_layer, accuracies


def score_candidates(
    samples: List[Sample],
    existing_scores: dict,
    tokenizer,
    model,
    direction: torch.Tensor,
    layer_idx: int,
    max_length: int,
    batch_size: int,
    output_cache: Path,
    overwrite: bool,
) -> dict:
    scores = {} if overwrite else dict(existing_scores)
    direction = direction.float().cpu()

    for sample in tqdm(samples, desc="Scoring candidates"):
        pending = []
        pending_indices = []
        for i, candidate in enumerate(sample.candidates):
            key = (sample.idx, i)
            if not candidate.compiles:
                continue
            if key in scores and not overwrite:
                continue
            pending.append(make_prompt(sample, candidate.code))
            pending_indices.append(i)

        if not pending:
            continue

        hidden = extract_hidden_states(pending, tokenizer, model, max_length, batch_size)
        layer_hidden = hidden[layer_idx]
        values = layer_hidden @ direction
        for cand_idx, value in zip(pending_indices, values.tolist()):
            scores[(sample.idx, cand_idx)] = {"repe_correctness": float(value)}

        tmp_path = output_cache.with_suffix(".tmp")
        with open(tmp_path, "wb") as handle:
            pickle.dump(scores, handle)
        os.replace(tmp_path, output_cache)

    return scores


def evaluate_repe_on_test(samples: List[Sample], repe_scores: dict, seed: int) -> dict:
    rng = random.Random(seed)
    result = {
        "repe_correctness": {"correct": 0, "total": 0},
        "random": {"correct": 0, "total": 0},
        "oracle": {"correct": 0, "total": 0},
    }

    for sample in samples:
        compiling = [c for c in sample.candidates if c.compiles]
        scored = [
            (i, c)
            for i, c in enumerate(sample.candidates)
            if c.compiles and (sample.idx, i) in repe_scores
        ]
        if not compiling or not scored:
            continue

        result["oracle"]["total"] += 1
        result["random"]["total"] += 1
        result["repe_correctness"]["total"] += 1

        if any(c.passes_tests for c in compiling):
            result["oracle"]["correct"] += 1

        if rng.choice(compiling).passes_tests:
            result["random"]["correct"] += 1

        _, best = max(scored, key=lambda item: repe_scores[(sample.idx, item[0])]["repe_correctness"])
        if best.passes_tests:
            result["repe_correctness"]["correct"] += 1

    return result


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_input_paths(args.cache_dir, args.candidates_file, args.seed, args.num_samples, args.num_candidates)
    seed = resolved["seed"]
    num_samples = resolved["num_samples"]
    rng = random.Random(seed)
    candidates_file = resolved["candidates_file"]
    compilation_file = resolved["compilation_file"]
    test_file = resolved["test_file"]
    output_cache = resolved["output_cache"]
    model_cache = resolved["model_cache"]
    results_file = resolved["results_file"]

    print(f"Loading candidates from {candidates_file}")
    samples: List[Sample] = load_pickle(candidates_file)
    compile_results = load_pickle(compilation_file)
    test_results = load_pickle(test_file)
    apply_cached_labels(samples, compile_results, test_results)

    fit_samples, val_samples, test_samples = split_samples(samples, args.fit_samples, args.val_samples)
    print(f"Split sizes: fit={len(fit_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    fit_pairs = build_pairs(fit_samples, args.max_pairs_per_split, rng)
    val_pairs = build_pairs(val_samples, args.max_pairs_per_split, rng)
    print(f"Pair counts: fit={len(fit_pairs)}, val={len(val_pairs)}")
    if not fit_pairs:
        raise RuntimeError("No fit pairs found. Need samples with both passing and failing compiling candidates.")
    if not val_pairs:
        raise RuntimeError("No validation pairs found. Need samples with both passing and failing compiling candidates.")

    tokenizer, model = load_model(args.model_name)

    fit_correct_prompts = [make_prompt(sample, good.code) for sample, good, _ in fit_pairs]
    fit_wrong_prompts = [make_prompt(sample, bad.code) for sample, _, bad in fit_pairs]
    val_correct_prompts = [make_prompt(sample, good.code) for sample, good, _ in val_pairs]
    val_wrong_prompts = [make_prompt(sample, bad.code) for sample, _, bad in val_pairs]

    print("Extracting fit hidden states")
    fit_correct_hidden = extract_hidden_states(fit_correct_prompts, tokenizer, model, args.max_length, args.batch_size)
    fit_wrong_hidden = extract_hidden_states(fit_wrong_prompts, tokenizer, model, args.max_length, args.batch_size)
    directions = fit_directions(fit_correct_hidden, fit_wrong_hidden)

    print("Extracting validation hidden states")
    val_correct_hidden = extract_hidden_states(val_correct_prompts, tokenizer, model, args.max_length, args.batch_size)
    val_wrong_hidden = extract_hidden_states(val_wrong_prompts, tokenizer, model, args.max_length, args.batch_size)
    best_layer, val_accuracies = choose_best_layer(directions, val_correct_hidden, val_wrong_hidden)

    print(f"Selected layer {best_layer} with validation pair accuracy {val_accuracies[best_layer]:.3f}")
    torch.save(
        {
            "dataset": "humaneval",
            "model_name": args.model_name,
            "best_layer": best_layer,
            "directions": directions,
            "val_accuracies": val_accuracies,
            "fit_samples": args.fit_samples,
            "val_samples": args.val_samples,
            "max_length": args.max_length,
        },
        model_cache,
    )
    print(f"Saved RepE model cache to {model_cache}")

    existing_scores = {}
    if output_cache.exists() and not args.overwrite_scores:
        existing_scores = load_pickle(output_cache)
        print(f"Loaded {len(existing_scores)} existing RepE scores from {output_cache}")

    repe_scores = score_candidates(
        samples,
        existing_scores,
        tokenizer,
        model,
        directions[best_layer],
        best_layer,
        args.max_length,
        args.batch_size,
        output_cache,
        args.overwrite_scores,
    )
    print(f"Saved RepE score cache to {output_cache}")

    test_result = evaluate_repe_on_test(test_samples, repe_scores, seed)
    test_result["config"] = {
        "seed": seed,
        "num_samples": num_samples,
        "num_candidates": args.num_candidates,
        "candidates_file": str(candidates_file),
        "compilation_file": str(compilation_file),
        "test_file": str(test_file),
    }
    with open(results_file, "w", encoding="utf-8") as handle:
        json.dump(test_result, handle, indent=2)
    print(json.dumps(test_result, indent=2))
    print(f"Saved test-split RepE result to {results_file}")


if __name__ == "__main__":
    main()
