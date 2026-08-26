#!/usr/bin/env python3
import argparse
import json
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import Levenshtein
from codebleu import calc_codebleu


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
    real_io_pairs: dict
    real_iospec: str
    candidates: List[Candidate] = field(default_factory=list)


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Sample":
            return Sample
        if name == "Candidate":
            return Candidate
        return super().find_class(module, name)


def load_samples(path: Path) -> List[Sample]:
    with path.open("rb") as f:
        return CompatUnpickler(f).load()


def load_pickle_dict(path: Path) -> Dict:
    with path.open("rb") as f:
        return pickle.load(f)


def infer_seed_tag(in_pkl: Path) -> str:
    match = re.search(r"(seed\d+)", in_pkl.name)
    if not match:
        raise ValueError(f"Could not infer seed tag from pickle name: {in_pkl.name}")
    return match.group(1)


def enrich_samples_from_phase_cache(samples: List[Sample], in_pkl: Path) -> Dict[str, object]:
    cache_dir = in_pkl.parent
    seed_tag = infer_seed_tag(in_pkl)
    compilation_path = cache_dir / f"phase1_compilation_{seed_tag}.pkl"
    tests_path = cache_dir / f"phase4_test_results_{seed_tag}.pkl"

    summary = {
        "used_phase_cache": False,
        "seed_tag": seed_tag,
        "phase1_compilation": str(compilation_path),
        "phase4_test_results": str(tests_path),
        "updated_compiles": 0,
        "updated_passes_tests": 0,
        "missing_compilation_entries": 0,
        "missing_test_entries": 0,
    }

    if not compilation_path.exists() or not tests_path.exists():
        return summary

    compilation = load_pickle_dict(compilation_path)
    test_results = load_pickle_dict(tests_path)

    for sample_idx, sample in enumerate(samples):
        for cand_idx, candidate in enumerate(sample.candidates):
            key = (sample_idx, cand_idx)
            if key in compilation:
                candidate.compiles = bool(compilation[key])
                summary["updated_compiles"] += 1
            else:
                summary["missing_compilation_entries"] += 1

            if key in test_results:
                candidate.passes_tests = bool(test_results[key])
                summary["updated_passes_tests"] += 1
            else:
                summary["missing_test_entries"] += 1

    summary["used_phase_cache"] = True
    return summary


def normalize_code(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = re.sub(r"\s+", " ", code)
    return code.strip()


def extract_function_body(code: str) -> str:
    lines = []
    for line in code.split("\n"):
        if line.strip().startswith("#include"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def compute_levenshtein_similarity(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    distance = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len)


def compute_codebleu(reference: str, candidate: str) -> float:
    try:
        if not reference.strip() or not candidate.strip():
            return 0.0
        result = calc_codebleu(
            references=[[reference]],
            predictions=[candidate],
            lang="c",
            weights=(0.25, 0.25, 0.25, 0.25),
            tokenizer=None,
        )
        return result["codebleu"]
    except Exception:
        return 0.0


def pick_most_frequent(candidates: List[Candidate], normalize: bool) -> Candidate:
    key_fn = normalize_code if normalize else (lambda s: s)
    counter = Counter(key_fn(c.code) for c in candidates)
    best_count = max(counter.values())
    best_keys = {k for k, v in counter.items() if v == best_count}

    for candidate in candidates:
        if key_fn(candidate.code) in best_keys:
            return candidate

    raise RuntimeError("No candidate selected from non-empty candidate list")


def evaluate(samples: List[Sample], compile_only: bool, normalize: bool) -> Dict[str, object]:
    total = 0
    correct = 0
    no_candidates = 0
    duplicate_winners = 0
    max_vote_hist = Counter()
    levenshtein_sum = 0.0
    codebleu_sum = 0.0
    metric_count = 0

    for sample in samples:
        pool = [c for c in sample.candidates if c.compiles] if compile_only else list(sample.candidates)
        if not pool:
            no_candidates += 1
            continue

        total += 1
        key_fn = normalize_code if normalize else (lambda s: s)
        votes = Counter(key_fn(c.code) for c in pool)
        winner_votes = max(votes.values())
        max_vote_hist[winner_votes] += 1
        if winner_votes > 1:
            duplicate_winners += 1

        chosen = pick_most_frequent(pool, normalize=normalize)
        original = extract_function_body(sample.c_func)
        selected = extract_function_body(chosen.code)
        levenshtein_sum += compute_levenshtein_similarity(original, selected)
        codebleu_sum += compute_codebleu(original, selected)
        metric_count += 1
        if chosen.passes_tests:
            correct += 1

    accuracy = (correct / total) if total else 0.0
    return {
        "correct": correct,
        "total": total,
        "no_candidates": no_candidates,
        "accuracy": accuracy,
        "duplicate_winner_samples": duplicate_winners,
        "max_vote_histogram": dict(sorted(max_vote_hist.items())),
        "levenshtein": (100.0 * levenshtein_sum / metric_count) if metric_count else 0.0,
        "codebleu": (100.0 * codebleu_sum / metric_count) if metric_count else 0.0,
    }


def selection_result_entry(method: str, stats: Dict[str, object]) -> Dict[str, object]:
    return {
        "method": method,
        "pass_rate": float(stats.get("accuracy", 0.0)) * 100.0,
        "correct": int(stats.get("correct", 0)),
        "total": int(stats.get("total", 0)),
        "levenshtein": float(stats.get("levenshtein", 0.0)),
        "codebleu": float(stats.get("codebleu", 0.0)),
    }


def find_base_selection_results(selection_results_path: Path) -> Optional[Path]:
    if selection_results_path.exists():
        return selection_results_path

    script_dir = Path(__file__).resolve().parent
    candidates = [
        selection_results_path,
        script_dir / selection_results_path.name,
        Path(__file__).resolve().parent.parent / selection_results_path.name,
        Path(__file__).resolve().parent.parent / "selection_evaluation_results.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def update_selection_results_file(selection_results_path: Path, results: Dict[str, object]) -> None:
    base_path = find_base_selection_results(selection_results_path)

    if base_path is not None:
        with base_path.open("r", encoding="utf-8") as f:
            selection_results = json.load(f)
    else:
        selection_results = {
            "config": {
                "source": "majority_vote_evaluation.py",
            },
            "results": [],
            "metrics_description": {
                "pass_rate": "Percentage of samples where selected candidate passes all tests",
                "levenshtein": "Average Levenshtein similarity to original source (%)",
                "codebleu": "Average CodeBLEU score comparing to original source (%)",
            },
        }

    existing = {
        item.get("method"): item
        for item in selection_results.get("results", [])
        if isinstance(item, dict) and "method" in item
    }

    for method in [
        "majority_vote_exact_all",
        "majority_vote_exact_compile_only",
        "majority_vote_normalized_all",
        "majority_vote_normalized_compile_only",
    ]:
        existing[method] = selection_result_entry(method, results[method])

    merged_results = list(existing.values())
    merged_results.sort(key=lambda x: x.get("pass_rate", 0.0), reverse=True)
    selection_results["results"] = merged_results

    with selection_results_path.open("w", encoding="utf-8") as f:
        json.dump(selection_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate majority-vote baselines from an RQ2 candidate pickle."
    )
    parser.add_argument(
        "--in-pkl",
        required=True,
        help="Input candidate pickle. Best used on the evaluated pickle where passes_tests is already populated.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Output JSON path (default: next to input pickle as majority_vote_results.json).",
    )
    parser.add_argument(
        "--selection-results",
        default=None,
        help="Path to selection_evaluation_results JSON to update. Default: inferred from seed, e.g. selection_evaluation_results_seed42.json",
    )
    args = parser.parse_args()

    in_pkl = Path(args.in_pkl)
    out_json = Path(args.out_json) if args.out_json else in_pkl.with_name("majority_vote_results.json")
    seed_tag = infer_seed_tag(in_pkl)
    default_selection_results = Path(f"selection_evaluation_results_{seed_tag}.json")
    selection_results_path = Path(args.selection_results) if args.selection_results else default_selection_results

    samples = load_samples(in_pkl)
    enrichment = enrich_samples_from_phase_cache(samples, in_pkl)

    results = {
        "input_pickle": str(in_pkl),
        "num_samples": len(samples),
        "phase_cache_enrichment": enrichment,
        "majority_vote_exact_all": evaluate(samples, compile_only=False, normalize=False),
        "majority_vote_exact_compile_only": evaluate(samples, compile_only=True, normalize=False),
        "majority_vote_normalized_all": evaluate(samples, compile_only=False, normalize=True),
        "majority_vote_normalized_compile_only": evaluate(samples, compile_only=True, normalize=True),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    update_selection_results_file(selection_results_path, results)

    print(f"Saved results to {out_json}")
    print(f"Updated selection results: {selection_results_path}")
    if enrichment["used_phase_cache"]:
        print(
            "Loaded phase caches: "
            f"compiles={enrichment['updated_compiles']}, "
            f"passes_tests={enrichment['updated_passes_tests']}, "
            f"missing_compiles={enrichment['missing_compilation_entries']}, "
            f"missing_tests={enrichment['missing_test_entries']}"
        )
    else:
        print("Phase caches not found; using candidate flags already stored in the pickle.")
    for key, stats in results.items():
        if not key.startswith("majority_vote_"):
            continue
        acc = 100 * stats["accuracy"]
        print(
            f"{key}: {stats['correct']}/{stats['total']} = {acc:.1f}% "
            f"(no candidates: {stats['no_candidates']}, duplicate winners: {stats['duplicate_winner_samples']})"
        )


if __name__ == "__main__":
    main()
