#!/usr/bin/env python3
"""
LLM4Decompile Selection Pipeline

Generates 200 candidates per sample, filters by compilation, ranks using
multiple similarity metrics, and evaluates pass@1 for each metric.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import subprocess
import tempfile
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter
from tqdm import tqdm
import re
import pickle
from pathlib import Path

# vLLM for fast inference
from vllm import LLM, SamplingParams

# tree-sitter for AST parsing
import tree_sitter_c as tsc
from tree_sitter import Language, Parser, Node

# Initialize tree-sitter
C_LANGUAGE = Language(tsc.language())
PARSER = Parser(C_LANGUAGE)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    model_name: str = "LLM4Binary/llm4decompile-9b-v2"
    num_samples: int = 656  # Number of test samples to evaluate (all samples)
    num_candidates: int = 200  # Candidates per sample
    temperatures: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
    max_new_tokens: int = 512
    cache_dir: str = "./cache"
    num_workers: int = 64  # For parallel compilation


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
# Tokenization
# ============================================================================

def tokenize_c_code(code: str) -> List[str]:
    """Simple C tokenizer using regex."""
    # Match identifiers, numbers, operators, punctuation
    pattern = r'[a-zA-Z_][a-zA-Z0-9_]*|0x[0-9a-fA-F]+|[0-9]+\.?[0-9]*|[+\-*/%=<>!&|^~?:;,.\[\]{}()]|"[^"]*"|\'[^\']*\''
    tokens = re.findall(pattern, code)
    return tokens


def get_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    """Extract n-grams from token list."""
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


# ============================================================================
# AST Analysis with tree-sitter
# ============================================================================

def parse_code(code: str) -> Optional[Node]:
    """Parse C code and return AST root."""
    try:
        tree = PARSER.parse(bytes(code, 'utf8'))
        return tree.root_node
    except:
        return None


def get_node_type_counts(node: Node) -> Counter:
    """Count occurrences of each node type in the AST."""
    counts = Counter()

    def traverse(n: Node):
        counts[n.type] += 1
        for child in n.children:
            traverse(child)

    if node:
        traverse(node)
    return counts


def get_ast_depth(node: Node) -> int:
    """Get maximum depth of AST."""
    if not node or not node.children:
        return 1
    return 1 + max(get_ast_depth(child) for child in node.children)


def get_control_flow_counts(node: Node) -> Dict[str, int]:
    """Count control flow constructs."""
    counts = {
        'if_statement': 0,
        'for_statement': 0,
        'while_statement': 0,
        'do_statement': 0,
        'return_statement': 0,
        'call_expression': 0,
        'binary_expression': 0,
        'assignment_expression': 0,
    }

    def traverse(n: Node):
        if n.type in counts:
            counts[n.type] += 1
        for child in n.children:
            traverse(child)

    if node:
        traverse(node)
    return counts


def get_identifiers(node: Node) -> Set[str]:
    """Extract all identifiers from AST."""
    identifiers = set()

    def traverse(n: Node):
        if n.type == 'identifier':
            identifiers.add(n.text.decode('utf8'))
        for child in n.children:
            traverse(child)

    if node:
        traverse(node)
    return identifiers


def get_function_signature(node: Node) -> Optional[Tuple[str, int]]:
    """Extract return type and parameter count from function definition."""
    if not node:
        return None

    for child in node.children:
        if child.type == 'function_definition':
            # Find declarator
            for c in child.children:
                if c.type == 'function_declarator':
                    # Count parameters
                    for cc in c.children:
                        if cc.type == 'parameter_list':
                            params = [p for p in cc.children if p.type == 'parameter_declaration']
                            return (child.children[0].text.decode('utf8') if child.children else '', len(params))
            break
    return None


# ============================================================================
# Similarity Metrics
# ============================================================================

def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def normalized_edit_similarity(code: str, ghidra: str) -> float:
    """Normalized edit distance similarity (1 = identical, 0 = very different)."""
    # Use only function body, skip boilerplate
    dist = edit_distance(code, ghidra)
    max_len = max(len(code), len(ghidra))
    if max_len == 0:
        return 1.0
    return 1.0 - (dist / max_len)


def token_f1_score(code: str, ghidra: str) -> float:
    """F1 score based on token overlap."""
    code_tokens = set(tokenize_c_code(code))
    ghidra_tokens = set(tokenize_c_code(ghidra))

    if not code_tokens or not ghidra_tokens:
        return 0.0

    intersection = code_tokens & ghidra_tokens
    precision = len(intersection) / len(code_tokens) if code_tokens else 0
    recall = len(intersection) / len(ghidra_tokens) if ghidra_tokens else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ngram_similarity(code: str, ghidra: str, n: int = 2) -> float:
    """Jaccard similarity of n-grams."""
    code_tokens = tokenize_c_code(code)
    ghidra_tokens = tokenize_c_code(ghidra)

    code_ngrams = get_ngrams(code_tokens, n)
    ghidra_ngrams = get_ngrams(ghidra_tokens, n)

    if not code_ngrams or not ghidra_ngrams:
        return 0.0

    intersection = code_ngrams & ghidra_ngrams
    union = code_ngrams | ghidra_ngrams

    return len(intersection) / len(union) if union else 0.0


def ast_node_similarity(code: str, ghidra: str) -> float:
    """Similarity based on AST node type distribution."""
    code_ast = parse_code(code)
    ghidra_ast = parse_code(ghidra)

    if not code_ast or not ghidra_ast:
        return 0.0

    code_counts = get_node_type_counts(code_ast)
    ghidra_counts = get_node_type_counts(ghidra_ast)

    # Cosine similarity of count vectors
    all_types = set(code_counts.keys()) | set(ghidra_counts.keys())

    dot_product = sum(code_counts.get(t, 0) * ghidra_counts.get(t, 0) for t in all_types)
    code_norm = sum(v**2 for v in code_counts.values()) ** 0.5
    ghidra_norm = sum(v**2 for v in ghidra_counts.values()) ** 0.5

    if code_norm == 0 or ghidra_norm == 0:
        return 0.0

    return dot_product / (code_norm * ghidra_norm)


def control_flow_similarity(code: str, ghidra: str) -> float:
    """Similarity based on control flow construct counts."""
    code_ast = parse_code(code)
    ghidra_ast = parse_code(ghidra)

    if not code_ast or not ghidra_ast:
        return 0.0

    code_cf = get_control_flow_counts(code_ast)
    ghidra_cf = get_control_flow_counts(ghidra_ast)

    # Compute similarity as 1 - normalized difference
    total_diff = 0
    total_max = 0

    for key in code_cf:
        total_diff += abs(code_cf[key] - ghidra_cf[key])
        total_max += max(code_cf[key], ghidra_cf[key])

    if total_max == 0:
        return 1.0

    return 1.0 - (total_diff / (2 * total_max))


def identifier_similarity(code: str, ghidra: str) -> float:
    """Jaccard similarity of identifiers."""
    code_ast = parse_code(code)
    ghidra_ast = parse_code(ghidra)

    if not code_ast or not ghidra_ast:
        return 0.0

    code_ids = get_identifiers(code_ast)
    ghidra_ids = get_identifiers(ghidra_ast)

    # Filter out very common identifiers
    common = {'int', 'char', 'void', 'long', 'float', 'double', 'return', 'if', 'else', 'for', 'while'}
    code_ids = code_ids - common
    ghidra_ids = ghidra_ids - common

    if not code_ids or not ghidra_ids:
        return 0.0

    intersection = code_ids & ghidra_ids
    union = code_ids | ghidra_ids

    return len(intersection) / len(union) if union else 0.0


def length_ratio_score(code: str, ghidra: str) -> float:
    """Score based on length ratio (penalize very different lengths)."""
    code_len = len(code.strip())
    ghidra_len = len(ghidra.strip())

    if ghidra_len == 0:
        return 0.0

    ratio = code_len / ghidra_len

    # Ideal ratio around 0.5-1.5 (generated C is usually shorter than Ghidra)
    # Penalize extremes
    if ratio < 0.2:
        return ratio / 0.2 * 0.5
    elif ratio > 3.0:
        return max(0, 1.0 - (ratio - 3.0) / 3.0) * 0.5
    else:
        # Score peaks around ratio 0.7
        return 1.0 - abs(ratio - 0.7) / 2.3


def compute_all_metrics(code: str, ghidra: str) -> Dict[str, float]:
    """Compute all similarity metrics."""
    return {
        'edit_similarity': normalized_edit_similarity(code, ghidra),
        'token_f1': token_f1_score(code, ghidra),
        'bigram_sim': ngram_similarity(code, ghidra, 2),
        'trigram_sim': ngram_similarity(code, ghidra, 3),
        'ast_node_sim': ast_node_similarity(code, ghidra),
        'control_flow_sim': control_flow_similarity(code, ghidra),
        'identifier_sim': identifier_similarity(code, ghidra),
        'length_ratio': length_ratio_score(code, ghidra),
    }


# ============================================================================
# Compilation and Evaluation
# ============================================================================

def check_compilation(args: Tuple[str, str, str]) -> Tuple[bool, str]:
    """Check if code compiles. Returns (compiles, error_message)."""
    code, c_func, c_test = args

    # Extract includes
    c_include = ''
    for line in c_func.split('\n'):
        if '#include' in line:
            c_include += line + '\n'
    for line in c_test.split('\n'):
        if '#include' in line:
            c_include += line + '\n'

    # Clean test of includes
    c_test_clean = '\n'.join(line for line in c_test.split('\n') if '#include' not in line)

    c_combine = c_include + '\n' + code + '\n' + c_test_clean

    with tempfile.TemporaryDirectory() as temp_dir:
        c_file = os.path.join(temp_dir, 'test.c')
        exe = os.path.join(temp_dir, 'test')

        with open(c_file, 'w') as f:
            f.write(c_combine)

        try:
            result = subprocess.run(
                f'gcc {c_file} -o {exe} -lm -w',
                shell=True, capture_output=True, timeout=10
            )
            return result.returncode == 0, result.stderr.decode()[:500]
        except subprocess.TimeoutExpired:
            return False, "Compilation timeout"
        except Exception as e:
            return False, str(e)


def run_test(args: Tuple[str, str, str]) -> bool:
    """Run compiled code and check if tests pass."""
    code, c_func, c_test = args

    # Extract includes
    c_include = ''
    for line in c_func.split('\n'):
        if '#include' in line:
            c_include += line + '\n'
    for line in c_test.split('\n'):
        if '#include' in line:
            c_include += line + '\n'

    c_test_clean = '\n'.join(line for line in c_test.split('\n') if '#include' not in line)
    c_combine = c_include + '\n' + code + '\n' + c_test_clean

    with tempfile.TemporaryDirectory() as temp_dir:
        c_file = os.path.join(temp_dir, 'test.c')
        exe = os.path.join(temp_dir, 'test')

        with open(c_file, 'w') as f:
            f.write(c_combine)

        try:
            result = subprocess.run(
                f'gcc {c_file} -o {exe} -lm -w',
                shell=True, capture_output=True, timeout=10
            )
            if result.returncode != 0:
                return False

            result = subprocess.run(
                exe, shell=True, capture_output=True, timeout=5
            )
            return result.returncode == 0
        except:
            return False


# ============================================================================
# Main Pipeline
# ============================================================================

def load_dataset(data_path: str) -> List[Dict]:
    """Load the evaluation dataset."""
    with open(data_path, 'r') as f:
        return json.load(f)


def generate_candidates(llm: LLM, samples: List[Sample], config: Config) -> List[Sample]:
    """Generate candidates for all samples."""
    print("\n=== Phase 1: Generating Candidates ===")

    # Filter samples that are too long
    tokenizer = llm.get_tokenizer()
    max_prompt_len = 8192 - config.max_new_tokens  # Leave room for generation

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

    # Use same params for batch generation (vLLM requirement)
    # We'll generate multiple batches with different temperatures
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

    # Assign candidates to valid samples
    for sample in valid_samples:
        sample.candidates = sample_to_candidates[sample.idx]

    # Return all samples (including skipped ones with empty candidates)
    return samples


def filter_by_compilation(samples: List[Sample], config: Config) -> List[Sample]:
    """Filter candidates that don't compile."""
    print("\n=== Phase 2: Filtering by Compilation ===")

    # Prepare all compilation tasks
    tasks = []
    task_info = []  # (sample_idx, candidate_idx)

    for sample in samples:
        for i, candidate in enumerate(sample.candidates):
            tasks.append((candidate.code, sample.c_func, sample.c_test))
            task_info.append((sample.idx, i))

    print(f"Checking {len(tasks)} candidates for compilation...")

    # Run in parallel
    with mp.Pool(config.num_workers) as pool:
        results = list(tqdm(pool.imap(check_compilation, tasks), total=len(tasks), desc="Compiling"))

    # Update candidates
    sample_dict = {s.idx: s for s in samples}
    for (sample_idx, cand_idx), (compiles, _) in zip(task_info, results):
        sample_dict[sample_idx].candidates[cand_idx].compiles = compiles

    # Report stats
    for sample in samples:
        compiling = sum(1 for c in sample.candidates if c.compiles)
        print(f"  Sample {sample.idx} ({sample.opt}): {compiling}/{len(sample.candidates)} compile")

    return samples


def compute_metrics(samples: List[Sample]) -> List[Sample]:
    """Compute similarity metrics for all compiling candidates."""
    print("\n=== Phase 3: Computing Metrics ===")

    for sample in tqdm(samples, desc="Samples"):
        for candidate in sample.candidates:
            if candidate.compiles:
                candidate.metrics = compute_all_metrics(candidate.code, sample.ghidra)

    return samples


def evaluate_selection(samples: List[Sample], config: Config) -> Dict[str, Dict]:
    """Evaluate pass@1 for each selection metric."""
    print("\n=== Phase 4: Evaluating Selection Metrics ===")

    metrics = ['edit_similarity', 'token_f1', 'bigram_sim', 'trigram_sim',
               'ast_node_sim', 'control_flow_sim', 'identifier_sim', 'length_ratio']

    results = {m: {'correct': 0, 'total': 0, 'no_candidates': 0} for m in metrics}
    results['random'] = {'correct': 0, 'total': 0, 'no_candidates': 0}
    results['oracle'] = {'correct': 0, 'total': 0, 'no_candidates': 0}  # Best possible

    for sample in tqdm(samples, desc="Evaluating"):
        compiling = [c for c in sample.candidates if c.compiles]

        if not compiling:
            for m in metrics:
                results[m]['no_candidates'] += 1
            results['random']['no_candidates'] += 1
            results['oracle']['no_candidates'] += 1
            continue

        # First, check which candidates pass tests
        test_tasks = [(c.code, sample.c_func, sample.c_test) for c in compiling]
        with mp.Pool(min(len(test_tasks), config.num_workers)) as pool:
            test_results = pool.map(run_test, test_tasks)

        for c, passes in zip(compiling, test_results):
            c.passes_tests = passes

        # Oracle: any passing candidate
        any_pass = any(c.passes_tests for c in compiling)
        results['oracle']['total'] += 1
        if any_pass:
            results['oracle']['correct'] += 1

        # Random selection
        import random
        random_candidate = random.choice(compiling)
        results['random']['total'] += 1
        if random_candidate.passes_tests:
            results['random']['correct'] += 1

        # Selection by each metric
        for metric in metrics:
            results[metric]['total'] += 1

            # Select candidate with highest metric score
            best = max(compiling, key=lambda c: c.metrics.get(metric, 0))
            if best.passes_tests:
                results[metric]['correct'] += 1

    return results


def main():
    config = Config()

    # Create cache directory
    Path(config.cache_dir).mkdir(exist_ok=True)

    print("=" * 70)
    print("LLM4DECOMPILE SELECTION PIPELINE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Samples: {config.num_samples}")
    print(f"  Candidates per sample: {config.num_candidates}")
    print(f"  Temperatures: {config.temperatures}")
    print(f"  Workers: {config.num_workers}")

    # Load dataset
    data_path = str(Path(__file__).resolve().parent / "../../LLM4Decompile/legacy-test/decompile-eval-executable-gcc-ghidra.json")
    dataset = load_dataset(data_path)

    # Create samples
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

    # Check for cached candidates
    cache_file = Path(config.cache_dir) / f"candidates_{config.num_samples}_{config.num_candidates}.pkl"

    if cache_file.exists():
        print(f"\nLoading cached candidates from {cache_file}")
        with open(cache_file, 'rb') as f:
            samples = pickle.load(f)
    else:
        # Load model
        print("\nLoading LLM...")
        llm = LLM(
            model=config.model_name,
            trust_remote_code=True,
            dtype='bfloat16',
            max_model_len=8192,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=2,  # Use both GPUs
        )

        # Generate candidates
        samples = generate_candidates(llm, samples, config)

        # Save cache
        print(f"\nSaving candidates to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(samples, f)

        # Free GPU memory
        del llm

    # Filter by compilation
    samples = filter_by_compilation(samples, config)

    # Compute metrics
    samples = compute_metrics(samples)

    # Evaluate
    results = evaluate_selection(samples, config)

    # Report results
    print("\n" + "=" * 70)
    print("RESULTS: PASS@1 BY SELECTION METRIC")
    print("=" * 70)

    # Sort by accuracy
    sorted_metrics = sorted(results.items(), key=lambda x: x[1]['correct']/max(x[1]['total'],1), reverse=True)

    for metric, stats in sorted_metrics:
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            print(f"  {metric:20s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:5.1f}%  (no candidates: {stats['no_candidates']})")

    # Save detailed results
    results_file = Path(config.cache_dir) / "selection_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Ensemble: combine best metrics
    print("\n" + "=" * 70)
    print("ENSEMBLE ANALYSIS")
    print("=" * 70)

    # Try combining top metrics
    top_metrics = ['control_flow_sim', 'ast_node_sim', 'token_f1']

    ensemble_correct = 0
    ensemble_total = 0

    for sample in samples:
        compiling = [c for c in sample.candidates if c.compiles]
        if not compiling:
            continue

        ensemble_total += 1

        # Weighted combination of top metrics
        def ensemble_score(c):
            return (c.metrics.get('control_flow_sim', 0) * 0.4 +
                    c.metrics.get('ast_node_sim', 0) * 0.3 +
                    c.metrics.get('token_f1', 0) * 0.3)

        best = max(compiling, key=ensemble_score)
        if best.passes_tests:
            ensemble_correct += 1

    if ensemble_total > 0:
        print(f"  Ensemble (CF+AST+TF1): {ensemble_correct}/{ensemble_total} = {100*ensemble_correct/ensemble_total:.1f}%")


if __name__ == "__main__":
    main()
