#!/usr/bin/env python3
"""
LLM4Decompile Selection Pipeline

Generates 200 candidates per sample, filters by compilation, ranks using
multiple similarity metrics, and evaluates pass@1 for each metric.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from concurrent.futures import ThreadPoolExecutor
# vLLM is only needed when regenerating candidates, not for cache-based MBR eval.
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

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
    num_candidates: int = 100  # Must match the shared RQ2 candidate cache
    temperatures: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
    max_new_tokens: int = 4096
    max_model_len: int = 12288
    cache_dir: str = "./cache"
    num_workers: int = 20  # For parallel compilation


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


def normalize_code_for_mbr(code: str) -> str:
    """Light normalization so consensus is less sensitive to formatting noise."""
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'\s+', ' ', code)
    return code.strip()


def multiset_f1(tokens1: List[str], tokens2: List[str]) -> float:
    """Token-overlap F1 that preserves multiplicity."""
    if not tokens1 or not tokens2:
        return 0.0

    counts1 = Counter(tokens1)
    counts2 = Counter(tokens2)
    overlap = sum(min(counts1[t], counts2[t]) for t in counts1.keys() | counts2.keys())

    precision = overlap / len(tokens1)
    recall = overlap / len(tokens2)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


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

        return compile_c_file(c_file, exe, timeout=10)


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
            compiled, _ = compile_c_file(c_file, exe, timeout=10)
            if not compiled:
                return False

            return run_binary(exe, timeout=5)
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
    if LLM is None or SamplingParams is None:
        raise RuntimeError("vLLM is not available in this environment.")

    print("\n=== Phase 1: Generating Candidates ===")

    # Filter samples that are too long
    tokenizer = llm.get_tokenizer()
    max_prompt_len = config.max_model_len - config.max_new_tokens  # Leave room for generation

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
    print("\n=== Phase 3: Computing Metrics (MBR Only) ===")

    for sample in tqdm(samples, desc="Samples"):
        compiling_cands = [c for c in sample.candidates if c.compiles]

        if not compiling_cands:
            continue

        mbr_scores = compute_mbr_scores(compiling_cands)
        for cand, score in zip(compiling_cands, mbr_scores):
            cand.metrics['mbr_consensus'] = score

    return samples


def evaluate_selection(samples: List[Sample], config: Config) -> Dict[str, Dict]:
    """Evaluate pass@1 for MBR against random and oracle baselines."""
    print("\n=== Phase 4: Evaluating MBR Selection ===")

    metrics = ['mbr_consensus']

    results = {'mbr_consensus': {'correct': 0, 'total': 0, 'no_candidates': 0}}
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
        # with mp.Pool(min(len(test_tasks), config.num_workers)) as pool:
        #     test_results = pool.map(run_test, test_tasks)
        with ThreadPoolExecutor(max_workers=20) as ex:
            test_results = list(ex.map(run_test, test_tasks))
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


# def compute_mbr_scores(candidates: List[Candidate]) -> List[float]:
#     """
#     Computes the MBR score for each candidate based on consensus.
#     High score = This candidate represents the 'average' or 'consensus' logic.
#     """
#     n = len(candidates)
#     if n <= 1: return [1.0] * n
    
#     # Pre-tokenize once to save 200x200 re-tokenization time
#     token_sets = [set(tokenize_c_code(c.code)) for c in candidates]
#     scores = []
    
#     for i in range(n):
#         total_similarity = 0.0
#         for j in range(n):
#             if i == j: continue
#             # Jaccard Similarity: Intersection / Union
#             inter = len(token_sets[i] & token_sets[j])
#             union = len(token_sets[i] | token_sets[j])
#             sim = inter / union if union > 0 else 0.0
#             total_similarity += sim
        
#         # Average similarity to all others
#         scores.append(total_similarity / (n - 1))
#     return scores

def compute_mbr_scores(candidates: List[Candidate]) -> List[float]:
    """
    Sample-based MBR over the candidate pool.

    The candidate list is treated as an empirical evidence set and each
    hypothesis is scored by its average utility against all sampled
    candidates. Duplicate samples naturally get more mass because they appear
    more often in the sample set; we do not apply an additional duplicate
    frequency weight.
    """
    n = len(candidates)
    if n <= 1:
        return [1.0] * n

    normalized_codes = [normalize_code_for_mbr(c.code) for c in candidates]

    token_lists = [tokenize_c_code(code) for code in normalized_codes]
    bigram_sets = [get_ngrams(tokens, 2) for tokens in token_lists]
    asts = [parse_code(code) for code in normalized_codes]
    node_counts = [get_node_type_counts(ast) if ast else Counter() for ast in asts]
    cf_counts = [get_control_flow_counts(ast) if ast else None for ast in asts]

    def jaccard(s1: Set[Tuple[str, ...]], s2: Set[Tuple[str, ...]]) -> float:
        if not s1 or not s2:
            return 0.0
        union = s1 | s2
        if not union:
            return 0.0
        return len(s1 & s2) / len(union)

    def cosine_counts(c1: Counter, c2: Counter) -> float:
        if not c1 or not c2:
            return 0.0
        keys = set(c1.keys()) | set(c2.keys())
        dot = sum(c1.get(k, 0) * c2.get(k, 0) for k in keys)
        norm1 = sum(v ** 2 for v in c1.values()) ** 0.5
        norm2 = sum(v ** 2 for v in c2.values()) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def control_flow_agreement(cf1: Optional[Dict[str, int]], cf2: Optional[Dict[str, int]]) -> float:
        if not cf1 or not cf2:
            return 0.0
        total_diff = 0
        total_max = 0
        for key in cf1:
            total_diff += abs(cf1[key] - cf2[key])
            total_max += max(cf1[key], cf2[key])
        if total_max == 0:
            return 1.0
        return 1.0 - (total_diff / (2 * total_max))

    pairwise_utility = [[0.0] * n for _ in range(n)]
    for i in range(n):
        pairwise_utility[i][i] = 1.0
        for j in range(i + 1, n):
            token_sim = multiset_f1(token_lists[i], token_lists[j])
            bigram_sim = jaccard(bigram_sets[i], bigram_sets[j])
            ast_sim = cosine_counts(node_counts[i], node_counts[j])
            cf_sim = control_flow_agreement(cf_counts[i], cf_counts[j])

            utility = (
                0.40 * token_sim +
                0.30 * bigram_sim +
                0.20 * ast_sim +
                0.10 * cf_sim
            )
            pairwise_utility[i][j] = utility
            pairwise_utility[j][i] = utility

    return [sum(row) / n for row in pairwise_utility]


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


def load_or_compute_mbr(samples: List[Sample], config: Config) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Load or compute only the MBR metric cache used by evaluate_selection.py."""
    mbr_cache = Path(config.cache_dir) / "phase2_5_mbr.pkl"

    if mbr_cache.exists():
        print("\n=== Phase 2: Loading cached MBR metrics ===")
        with open(mbr_cache, 'rb') as f:
            mbr_results = pickle.load(f)
    else:
        print("\n=== Phase 2: Computing MBR metrics ===")
        mbr_results = {}
        for sample in tqdm(samples, desc="MBR"):
            compiling = [(i, c) for i, c in enumerate(sample.candidates) if c.compiles]
            if not compiling:
                continue

            candidate_indices = [i for i, _ in compiling]
            candidate_objs = [c for _, c in compiling]
            mbr_scores = compute_mbr_scores(candidate_objs)

            for cand_idx, score in zip(candidate_indices, mbr_scores):
                mbr_results[(sample.idx, cand_idx)] = {'mbr_consensus': score}

        with open(mbr_cache, 'wb') as f:
            pickle.dump(mbr_results, f)
        print(f"Saved MBR cache to {mbr_cache}")

    for sample in samples:
        for i, candidate in enumerate(sample.candidates):
            metrics = mbr_results.get((sample.idx, i))
            if metrics:
                candidate.metrics.update(metrics)

    return mbr_results


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

def main():
    config = Config()

    # Create cache directory
    Path(config.cache_dir).mkdir(exist_ok=True)

    print("=" * 70)
    print("RQ2b ADVANCED RERANKING: MBR")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Samples: {config.num_samples}")
    print(f"  Candidates per sample: {config.num_candidates}")
    print(f"  Temperatures: {config.temperatures}")
    print(f"  Max new tokens: {config.max_new_tokens}")
    print(f"  Max model length: {config.max_model_len}")
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
            max_model_len=config.max_model_len,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,
        )

        # Generate candidates
        samples = generate_candidates(llm, samples, config)

        # Save cache
        print(f"\nSaving candidates to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(samples, f)

        # Free GPU memory
        del llm

    load_or_compute_compilation(samples, config)
    load_or_compute_mbr(samples, config)
    load_or_compute_test_results(samples, config)

    # Evaluate
    results = evaluate_selection(samples, config)

    # Report results
    print("\n" + "=" * 70)
    print(f"{'METRIC':<25} | {'PASS@1':<10} | {'SUCCESS/TOTAL'}")
    print("=" * 70)

    # Sort by accuracy
    sorted_metrics = sorted(results.items(), key=lambda x: x[1]['correct']/max(x[1]['total'],1), reverse=True)

    for metric, stats in sorted_metrics:
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            print(f"  {metric:20s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:5.1f}%  (no candidates: {stats['no_candidates']})")

    # Save detailed results
    results_file = Path(config.cache_dir) / "selection_results_mbr.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    print("=" * 70)


if __name__ == "__main__":
    main()
