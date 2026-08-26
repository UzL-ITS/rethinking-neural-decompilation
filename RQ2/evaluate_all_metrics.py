#!/usr/bin/env python3
"""
Comprehensive evaluation of all selection metrics including:
- Text-based metrics (edit distance, token F1, n-gram similarity)
- AST-based metrics (node similarity, control flow, identifiers)
- UniXcoder baseline (pretrained)
- UniXcoder finetuned (contrastive learning)
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
import random
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import UniXcoder
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from unixcoder import UniXcoder

# tree-sitter for AST parsing
import tree_sitter_c as tsc
from tree_sitter import Language, Parser, Node

# Initialize tree-sitter
C_LANGUAGE = Language(tsc.language())
PARSER = Parser(C_LANGUAGE)


# ============================================================================
# Configuration
# ============================================================================

# TrainConfig needed to unpickle checkpoint (must match training script)
@dataclass
class TrainConfig:
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
class Config:
    num_samples: int = 656
    num_candidates: int = 100
    cache_dir: str = "./cache"
    num_workers: int = 20
    unixcoder_base: str = "microsoft/unixcoder-base"
    unixcoder_finetuned: str = "./checkpoints/contrastive_unixcoder_best.pt"
    batch_size: int = 32
    llm4decompile_model: str = "LLM4Binary/llm4decompile-9b-v2"
    seed: Optional[int] = 42
    incremental_save_every: int = 100


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


def seed_suffix(seed: Optional[int]) -> str:
    return "" if seed is None else f"_seed{seed}"


def resolve_cache_file(cache_dir: Path, base_name: str, suffix: str, seed: Optional[int], required: bool = False) -> Path:
    preferred = cache_dir / f"{base_name}{seed_suffix(seed)}{suffix}"
    legacy = cache_dir / f"{base_name}{suffix}"

    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy

    if seed is None:
        matches = sorted(cache_dir.glob(f"{base_name}_seed*{suffix}"))
        if len(matches) == 1:
            return matches[0]

    if required:
        return preferred
    return preferred


def load_pickle_dict_if_exists(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data if isinstance(data, dict) else {}


def save_pickle_dict(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp_path, path)


# ============================================================================
# UniXcoder Models
# ============================================================================

class ContrastiveUniXcoder(nn.Module):
    """UniXcoder with projection head (must match training architecture)."""

    def __init__(self, model_name: str, projection_dim: int = 256):
        super().__init__()
        self.encoder = UniXcoder(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim),
        )

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, sentence_embeddings = self.encoder(input_ids)
        projected = self.projection(sentence_embeddings)
        return F.normalize(projected, p=2, dim=-1)


class UniXcoderEmbedder:
    """Wrapper for UniXcoder embeddings."""

    def __init__(self, model_name: str, checkpoint_path: str = None, projection_dim: int = 256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading finetuned UniXcoder from {checkpoint_path}")
            self.model = ContrastiveUniXcoder(model_name, projection_dim)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.is_finetuned = True
        else:
            print(f"Loading base UniXcoder from {model_name}")
            self.model = UniXcoder(model_name)
            self.is_finetuned = False

        self.model.eval()
        self.model = self.model.to(self.device)

    def get_embeddings(self, codes: List[str], batch_size: int = 32) -> torch.Tensor:
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(codes), batch_size):
                batch = codes[i:i+batch_size]

                if self.is_finetuned:
                    tokens_ids = self.model.encoder.tokenize(batch, max_length=512, padding=True, mode="<encoder-only>")
                    source_ids = torch.tensor(tokens_ids).to(self.device)
                    embeddings = self.model.encode(source_ids)
                else:
                    tokens_ids = self.model.tokenize(batch, max_length=512, padding=True, mode="<encoder-only>")
                    source_ids = torch.tensor(tokens_ids).to(self.device)
                    _, embeddings = self.model(source_ids)
                    embeddings = F.normalize(embeddings, p=2, dim=-1)

                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()


# ============================================================================
# Tokenization and AST
# ============================================================================

def tokenize_c_code(code: str) -> List[str]:
    pattern = r'[a-zA-Z_][a-zA-Z0-9_]*|0x[0-9a-fA-F]+|[0-9]+\.?[0-9]*|[+\-*/%=<>!&|^~?:;,.\[\]{}()]|"[^"]*"|\'[^\']*\''
    return re.findall(pattern, code)


def get_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def normalize_code_for_mbr(code: str) -> str:
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'\s+', ' ', code)
    return code.strip()


def parse_code(code: str) -> Optional[Node]:
    try:
        tree = PARSER.parse(bytes(code, 'utf8'))
        return tree.root_node
    except:
        return None


def get_node_type_counts(node: Node) -> Counter:
    counts = Counter()
    def traverse(n: Node):
        counts[n.type] += 1
        for child in n.children:
            traverse(child)
    if node:
        traverse(node)
    return counts


def get_control_flow_counts(node: Node) -> Dict[str, int]:
    counts = {'if_statement': 0, 'for_statement': 0, 'while_statement': 0,
              'do_statement': 0, 'return_statement': 0, 'call_expression': 0,
              'binary_expression': 0, 'assignment_expression': 0}
    def traverse(n: Node):
        if n.type in counts:
            counts[n.type] += 1
        for child in n.children:
            traverse(child)
    if node:
        traverse(node)
    return counts


def get_identifiers(node: Node) -> Set[str]:
    identifiers = set()
    def traverse(n: Node):
        if n.type == 'identifier':
            identifiers.add(n.text.decode('utf8'))
        for child in n.children:
            traverse(child)
    if node:
        traverse(node)
    return identifiers


# ============================================================================
# Similarity Metrics
# ============================================================================

def edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            curr_row.append(min(prev_row[j + 1] + 1, curr_row[j] + 1, prev_row[j] + (c1 != c2)))
        prev_row = curr_row
    return prev_row[-1]


def normalized_edit_similarity(code: str, ghidra: str) -> float:
    dist = edit_distance(code, ghidra)
    max_len = max(len(code), len(ghidra))
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0


def token_f1_score(code: str, ghidra: str) -> float:
    code_tokens = set(tokenize_c_code(code))
    ghidra_tokens = set(tokenize_c_code(ghidra))
    if not code_tokens or not ghidra_tokens:
        return 0.0
    intersection = code_tokens & ghidra_tokens
    precision = len(intersection) / len(code_tokens)
    recall = len(intersection) / len(ghidra_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def ngram_similarity(code: str, ghidra: str, n: int = 2) -> float:
    code_ngrams = get_ngrams(tokenize_c_code(code), n)
    ghidra_ngrams = get_ngrams(tokenize_c_code(ghidra), n)
    if not code_ngrams or not ghidra_ngrams:
        return 0.0
    intersection = code_ngrams & ghidra_ngrams
    union = code_ngrams | ghidra_ngrams
    return len(intersection) / len(union) if union else 0.0


def ast_node_similarity(code: str, ghidra: str) -> float:
    code_ast, ghidra_ast = parse_code(code), parse_code(ghidra)
    if not code_ast or not ghidra_ast:
        return 0.0
    code_counts = get_node_type_counts(code_ast)
    ghidra_counts = get_node_type_counts(ghidra_ast)
    all_types = set(code_counts.keys()) | set(ghidra_counts.keys())
    dot_product = sum(code_counts.get(t, 0) * ghidra_counts.get(t, 0) for t in all_types)
    code_norm = sum(v**2 for v in code_counts.values()) ** 0.5
    ghidra_norm = sum(v**2 for v in ghidra_counts.values()) ** 0.5
    return dot_product / (code_norm * ghidra_norm) if code_norm > 0 and ghidra_norm > 0 else 0.0


def control_flow_similarity(code: str, ghidra: str) -> float:
    code_ast, ghidra_ast = parse_code(code), parse_code(ghidra)
    if not code_ast or not ghidra_ast:
        return 0.0
    code_cf = get_control_flow_counts(code_ast)
    ghidra_cf = get_control_flow_counts(ghidra_ast)
    total_diff = sum(abs(code_cf[k] - ghidra_cf[k]) for k in code_cf)
    total_max = sum(max(code_cf[k], ghidra_cf[k]) for k in code_cf)
    return 1.0 - (total_diff / (2 * total_max)) if total_max > 0 else 1.0


def identifier_similarity(code: str, ghidra: str) -> float:
    code_ast, ghidra_ast = parse_code(code), parse_code(ghidra)
    if not code_ast or not ghidra_ast:
        return 0.0
    common = {'int', 'char', 'void', 'long', 'float', 'double', 'return', 'if', 'else', 'for', 'while'}
    code_ids = get_identifiers(code_ast) - common
    ghidra_ids = get_identifiers(ghidra_ast) - common
    if not code_ids or not ghidra_ids:
        return 0.0
    intersection = code_ids & ghidra_ids
    union = code_ids | ghidra_ids
    return len(intersection) / len(union) if union else 0.0


def compute_text_metrics(code: str, ghidra: str) -> Dict[str, float]:
    return {
        'edit_similarity': normalized_edit_similarity(code, ghidra),
        'token_f1': token_f1_score(code, ghidra),
        'bigram_sim': ngram_similarity(code, ghidra, 2),
        'trigram_sim': ngram_similarity(code, ghidra, 3),
        'ast_node_sim': ast_node_similarity(code, ghidra),
        'control_flow_sim': control_flow_similarity(code, ghidra),
        'identifier_sim': identifier_similarity(code, ghidra),
    }


def multiset_f1(tokens1: List[str], tokens2: List[str]) -> float:
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


def compute_mbr_scores(candidates: List[Candidate]) -> List[float]:
    n = len(candidates)
    if n <= 1:
        return [1.0] * n

    normalized_codes = [normalize_code_for_mbr(c.code) for c in candidates]
    normalized_counts = Counter(normalized_codes)
    probs = [normalized_counts[code] / n for code in normalized_codes]

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
            utility = 0.40 * token_sim + 0.30 * bigram_sim + 0.20 * ast_sim + 0.10 * cf_sim
            pairwise_utility[i][j] = utility
            pairwise_utility[j][i] = utility

    scores = []
    for i in range(n):
        expected_utility = 0.0
        for j in range(n):
            expected_utility += probs[j] * pairwise_utility[i][j]
        scores.append(expected_utility)
    return scores


def derive_execution_checkpoints(c_test: str, max_checks: int = 3) -> List[str]:
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


def compute_execution_guided_score(code: str, c_func: str, c_test: str, max_checks: int = 3) -> float:
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
    return compute_execution_guided_score(*args)


# ============================================================================
# Perplexity Computation
# ============================================================================

def create_llm4decompile_prompt(ghidra_code: str) -> str:
    """Create the standard LLM4Decompile prompt."""
    return (
        "# This is the assembly code:\n"
        f"{ghidra_code}\n"
        "# What is the source code?\n"
    )


class PerplexityComputer:
    """Compute perplexity for (ghidra, decompiled) pairs using LLM4Decompile."""

    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading LLM4Decompile model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("LLM4Decompile model loaded")

    def compute_perplexity(self, ghidra_code: str, decompiled_code: str) -> float:
        """
        Compute perplexity of decompiled code given ghidra prompt.
        Returns perplexity (lower = better fit).
        """
        try:
            prompt = create_llm4decompile_prompt(ghidra_code)
            full_text = prompt + decompiled_code

            # Tokenize
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
            full_tokens = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096)

            prompt_len = prompt_tokens.input_ids.shape[1]
            full_len = full_tokens.input_ids.shape[1]
            decompiled_len = full_len - prompt_len

            if decompiled_len < 5:
                return float('inf')

            input_ids = full_tokens.input_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits

            # Get logits for decompiled portion
            target_logits = logits[0, prompt_len-1:full_len-1, :]
            target_tokens = input_ids[0, prompt_len:full_len]

            # Compute log softmax and get log prob of actual tokens
            log_probs = F.log_softmax(target_logits, dim=-1)
            token_log_probs = log_probs.gather(1, target_tokens.unsqueeze(1)).squeeze(1)

            # Perplexity = exp(-average log prob)
            avg_log_prob = token_log_probs.float().mean().item()
            perplexity = float(torch.exp(torch.tensor(-avg_log_prob)).item())

            return perplexity

        except Exception as e:
            print(f"Error computing perplexity: {e}")
            return float('inf')


# ============================================================================
# Compilation and Testing
# ============================================================================

def check_compilation(args: Tuple[str, str, str]) -> bool:
    code, c_func, c_test = args
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
            result = subprocess.run(f'gcc {c_file} -o {exe} -lm -w', shell=True, capture_output=True, timeout=10)
            return result.returncode == 0
        except:
            return False


def run_test(args: Tuple[str, str, str]) -> bool:
    code, c_func, c_test = args
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
            result = subprocess.run(f'gcc {c_file} -o {exe} -lm -w', shell=True, capture_output=True, timeout=10)
            if result.returncode != 0:
                return False
            # print("running in sandbox")
            result = subprocess.run(f"./run-sandboxed.sh {exe}", shell=True, capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    config = Config()
    cache_dir = Path(config.cache_dir)

    if config.seed is not None:
        random.seed(config.seed)

    print("=" * 70)
    print("COMPREHENSIVE SELECTION METRIC EVALUATION")
    print("=" * 70)

    # Load cached candidates
    cache_file = resolve_cache_file(
        cache_dir,
        f"candidates_{config.num_samples}_{config.num_candidates}",
        ".pkl",
        config.seed,
        required=True,
    )
    if not cache_file.exists():
        print(f"ERROR: Cache file not found: {cache_file}")
        return

    print(f"\nLoading cached candidates from {cache_file}")
    with open(cache_file, 'rb') as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} samples")

    # ========================================================================
    # Phase 1: Filter by compilation (with caching)
    # ========================================================================
    compile_cache = resolve_cache_file(cache_dir, "phase1_compilation", ".pkl", config.seed)

    print("\n=== Phase 1: Filtering by Compilation ===")
    compile_results = load_pickle_dict_if_exists(compile_cache)
    if compile_results:
        print(f"Loaded {len(compile_results)} cached compilation entries from {compile_cache}")
    for sample in samples:
        for i, candidate in enumerate(sample.candidates):
            key = (sample.idx, i)
            if key in compile_results:
                candidate.compiles = compile_results[key]

    tasks = []
    task_info = []
    for sample in samples:
        for i, candidate in enumerate(sample.candidates):
            key = (sample.idx, i)
            if key in compile_results:
                continue
            tasks.append((candidate.code, sample.c_func, sample.c_test))
            task_info.append((sample.idx, i))

    if tasks:
        print(f"Checking {len(tasks)} remaining candidates for compilation...")
        with mp.Pool(config.num_workers) as pool:
            results = list(tqdm(pool.imap(check_compilation, tasks), total=len(tasks), desc="Compiling"))

        sample_dict = {s.idx: s for s in samples}
        for (sample_idx, cand_idx), compiles in zip(task_info, results):
            sample_dict[sample_idx].candidates[cand_idx].compiles = compiles
            compile_results[(sample_idx, cand_idx)] = compiles

        save_pickle_dict(compile_cache, compile_results)
        print(f"Saved compilation cache to {compile_cache}")

    total_compiling = sum(sum(1 for c in s.candidates if c.compiles) for s in samples)
    print(f"Total compiling candidates: {total_compiling}")

    # ========================================================================
    # Phase 2: Compute text-based metrics (with caching)
    # ========================================================================
    text_cache = resolve_cache_file(cache_dir, "phase2_text_metrics", ".pkl", config.seed)

    print("\n=== Phase 2: Computing Text Metrics ===")
    text_results = load_pickle_dict_if_exists(text_cache)
    if text_results:
        print(f"Loaded {len(text_results)} cached text metric entries from {text_cache}")
    for sample in samples:
        for i, candidate in enumerate(sample.candidates):
            key = (sample.idx, i)
            if key in text_results:
                candidate.metrics.update(text_results[key])

    dirty_count = 0
    for sample in tqdm(samples, desc="Text metrics"):
        for i, candidate in enumerate(sample.candidates):
            if not candidate.compiles:
                continue
            key = (sample.idx, i)
            if key in text_results:
                continue
            metrics = compute_text_metrics(candidate.code, sample.ghidra)
            candidate.metrics.update(metrics)
            text_results[key] = metrics
            dirty_count += 1
        if dirty_count and dirty_count % config.incremental_save_every == 0:
            save_pickle_dict(text_cache, text_results)
    save_pickle_dict(text_cache, text_results)
    print(f"Saved text metrics cache to {text_cache}")

    # ========================================================================
    # Phase 2.5: Compute MBR scores (with caching)
    # ========================================================================
    mbr_cache = resolve_cache_file(cache_dir, "phase2_5_mbr", ".pkl", config.seed)

    print("\n=== Phase 2.5: Computing MBR Metrics ===")
    mbr_results = load_pickle_dict_if_exists(mbr_cache)
    if mbr_results:
        print(f"Loaded {len(mbr_results)} cached MBR entries from {mbr_cache}")
    for sample in samples:
        for i, candidate in enumerate(sample.candidates):
            key = (sample.idx, i)
            if key in mbr_results:
                candidate.metrics.update(mbr_results[key])

    dirty_count = 0
    for sample in tqdm(samples, desc="MBR"):
        compiling = [(i, c) for i, c in enumerate(sample.candidates) if c.compiles]
        if not compiling:
            continue
        candidate_indices = [i for i, _ in compiling]
        if all((sample.idx, cand_idx) in mbr_results for cand_idx in candidate_indices):
            continue

        candidate_objs = [c for _, c in compiling]
        mbr_scores = compute_mbr_scores(candidate_objs)

        for cand_idx, candidate, score in zip(candidate_indices, candidate_objs, mbr_scores):
            key = (sample.idx, cand_idx)
            if key in mbr_results:
                continue
            metrics = {'mbr_consensus': score}
            candidate.metrics.update(metrics)
            mbr_results[key] = metrics
            dirty_count += 1

        if dirty_count and dirty_count % config.incremental_save_every == 0:
            save_pickle_dict(mbr_cache, mbr_results)

    save_pickle_dict(mbr_cache, mbr_results)
    print(f"Saved MBR cache to {mbr_cache}")

    # ========================================================================
    # Phase 3: Compute UniXcoder embeddings (with caching)
    # ========================================================================
    unixcoder_cache = resolve_cache_file(cache_dir, "phase3_unixcoder", ".pkl", config.seed)

    print("\n=== Phase 3: Computing UniXcoder Embeddings ===")
    unixcoder_results = load_pickle_dict_if_exists(unixcoder_cache)
    if unixcoder_results:
        print(f"Loaded {len(unixcoder_results)} cached UniXcoder entries from {unixcoder_cache}")
    for sample in samples:
        for i, candidate in enumerate(sample.candidates):
            key = (sample.idx, i)
            if key in unixcoder_results:
                candidate.metrics.update(unixcoder_results[key])

    print("\nLoading base UniXcoder...")
    embedder_base = UniXcoderEmbedder(config.unixcoder_base)
    print("\nLoading finetuned UniXcoder...")
    embedder_finetuned = UniXcoderEmbedder(config.unixcoder_base, config.unixcoder_finetuned)

    print("\nEmbedding Ghidra codes...")
    ghidra_codes = [s.ghidra for s in samples]
    ghidra_emb_base = embedder_base.get_embeddings(ghidra_codes, config.batch_size)
    ghidra_emb_ft = embedder_finetuned.get_embeddings(ghidra_codes, config.batch_size)

    print("\nComputing UniXcoder similarities for candidates...")
    dirty_count = 0
    for sample_idx, sample in enumerate(tqdm(samples, desc="UniXcoder")):
        compiling = [(i, c) for i, c in enumerate(sample.candidates) if c.compiles]
        if not compiling:
            continue

        missing = [(i, c) for i, c in compiling if (sample.idx, i) not in unixcoder_results]
        if not missing:
            continue

        candidate_codes = [c.code for _, c in missing]
        cand_emb_base = embedder_base.get_embeddings(candidate_codes, config.batch_size)
        cand_emb_ft = embedder_finetuned.get_embeddings(candidate_codes, config.batch_size)

        for j, (i, candidate) in enumerate(missing):
            metrics = {
                'unixcoder_base': embedder_base.cosine_similarity(
                    cand_emb_base[j], ghidra_emb_base[sample_idx]),
                'unixcoder_finetuned': embedder_finetuned.cosine_similarity(
                    cand_emb_ft[j], ghidra_emb_ft[sample_idx])
            }
            candidate.metrics.update(metrics)
            unixcoder_results[(sample.idx, i)] = metrics
            dirty_count += 1

        if dirty_count and dirty_count % config.incremental_save_every == 0:
            save_pickle_dict(unixcoder_cache, unixcoder_results)

    save_pickle_dict(unixcoder_cache, unixcoder_results)
    print(f"Saved UniXcoder cache to {unixcoder_cache}")

    # ========================================================================
    # Phase 3.5: Compute execution-guided scores (with caching)
    # ========================================================================
    execution_guided_cache = resolve_cache_file(cache_dir, "phase3_5_execution_guided", ".pkl", config.seed)

    print("\n=== Phase 3.5: Computing Execution-Guided Metrics ===")
    execution_guided_results = load_pickle_dict_if_exists(execution_guided_cache)
    if execution_guided_results:
        print(f"Loaded {len(execution_guided_results)} cached execution-guided entries from {execution_guided_cache}")
    for sample in samples:
        for i, candidate in enumerate(sample.candidates):
            key = (sample.idx, i)
            if key in execution_guided_results:
                candidate.metrics.update(execution_guided_results[key])

    dirty_count = 0
    for sample in tqdm(samples, desc="Execution-guided"):
        compiling = [(i, c) for i, c in enumerate(sample.candidates) if c.compiles]
        if not compiling:
            continue

        missing = [(i, c) for i, c in compiling if (sample.idx, i) not in execution_guided_results]
        if not missing:
            continue

        candidate_indices = [i for i, _ in missing]
        candidate_objs = [c for _, c in missing]
        tasks = [(c.code, sample.c_func, sample.c_test, 3) for c in candidate_objs]

        with ThreadPoolExecutor(max_workers=max(1, min(len(tasks), config.num_workers))) as ex:
            scores = list(ex.map(compute_execution_guided_score_from_args, tasks))

        for cand_idx, candidate, score in zip(candidate_indices, candidate_objs, scores):
            metrics = {'execution_guided': score}
            candidate.metrics.update(metrics)
            execution_guided_results[(sample.idx, cand_idx)] = metrics
            dirty_count += 1

        if dirty_count and dirty_count % config.incremental_save_every == 0:
            save_pickle_dict(execution_guided_cache, execution_guided_results)

    save_pickle_dict(execution_guided_cache, execution_guided_results)
    print(f"Saved execution-guided cache to {execution_guided_cache}")

    # ========================================================================
    # Phase 3.6: Compute Perplexity (with caching)
    # ========================================================================
    perplexity_cache = resolve_cache_file(cache_dir, "phase3_perplexity", ".pkl", config.seed)

    print("\n=== Phase 3.6: Computing Perplexity Metrics ===")
    perplexity_results = load_pickle_dict_if_exists(perplexity_cache)
    if perplexity_results:
        print(f"Loaded {len(perplexity_results)} cached perplexity entries from {perplexity_cache}")
    for sample in samples:
        for i, candidate in enumerate(sample.candidates):
            key = (sample.idx, i)
            if key in perplexity_results:
                candidate.metrics.update(perplexity_results[key])

    perplexity_computer = PerplexityComputer(config.llm4decompile_model)
    dirty_count = 0
    for sample in tqdm(samples, desc="Perplexity"):
        compiling = [(i, c) for i, c in enumerate(sample.candidates) if c.compiles]
        if not compiling:
            continue

        for i, candidate in compiling:
            key = (sample.idx, i)
            if key in perplexity_results:
                continue
            ppl = perplexity_computer.compute_perplexity(sample.ghidra, candidate.code)
            metrics = {
                'perplexity': ppl,
                'neg_perplexity': -ppl,
            }
            candidate.metrics.update(metrics)
            perplexity_results[key] = metrics
            dirty_count += 1

        if dirty_count and dirty_count % config.incremental_save_every == 0:
            save_pickle_dict(perplexity_cache, perplexity_results)

    save_pickle_dict(perplexity_cache, perplexity_results)
    print(f"Saved perplexity cache to {perplexity_cache}")

    # ========================================================================
    # Phase 4: Run tests (with caching)
    # ========================================================================
    test_cache = resolve_cache_file(cache_dir, "phase4_test_results", ".pkl", config.seed)

    print("\n=== Phase 4: Running Tests ===")
    test_results_cache = load_pickle_dict_if_exists(test_cache)
    if test_results_cache:
        print(f"Loaded {len(test_results_cache)} cached test entries from {test_cache}")
    for sample in samples:
        for i, candidate in enumerate(sample.candidates):
            key = (sample.idx, i)
            if key in test_results_cache:
                candidate.passes_tests = test_results_cache[key]

    dirty_count = 0
    for sample in tqdm(samples, desc="Testing"):
        compiling = [(i, c) for i, c in enumerate(sample.candidates) if c.compiles]
        if not compiling:
            continue

        missing = [(i, c) for i, c in compiling if (sample.idx, i) not in test_results_cache]
        if not missing:
            continue

        test_tasks = [(c.code, sample.c_func, sample.c_test) for _, c in missing]
        with mp.Pool(min(len(test_tasks), config.num_workers)) as pool:
            test_results = pool.map(run_test, test_tasks)

        for (i, c), passes in zip(missing, test_results):
            c.passes_tests = passes
            test_results_cache[(sample.idx, i)] = passes
            dirty_count += 1

        if dirty_count and dirty_count % config.incremental_save_every == 0:
            save_pickle_dict(test_cache, test_results_cache)

    save_pickle_dict(test_cache, test_results_cache)
    print(f"Saved test results cache to {test_cache}")

    # ========================================================================
    # Phase 5: Compute selection accuracy
    # ========================================================================
    print("\n=== Phase 5: Evaluating Selection ===")

    metrics = ['mbr_consensus', 'execution_guided',
               'edit_similarity', 'token_f1', 'bigram_sim', 'trigram_sim',
               'ast_node_sim', 'control_flow_sim', 'identifier_sim',
               'unixcoder_base', 'unixcoder_finetuned', 'neg_perplexity']

    results = {m: {'correct': 0, 'total': 0} for m in metrics}
    results['random'] = {'correct': 0, 'total': 0}
    results['oracle'] = {'correct': 0, 'total': 0}
    results['ensemble_text'] = {'correct': 0, 'total': 0}
    results['ensemble_unixcoder_ft'] = {'correct': 0, 'total': 0}
    results['ensemble_perplexity'] = {'correct': 0, 'total': 0}

    for sample in tqdm(samples, desc="Evaluating"):
        compiling = [c for c in sample.candidates if c.compiles]
        if not compiling:
            continue

        # Oracle
        any_pass = any(c.passes_tests for c in compiling)
        results['oracle']['total'] += 1
        if any_pass:
            results['oracle']['correct'] += 1

        # Random
        results['random']['total'] += 1
        if random.choice(compiling).passes_tests:
            results['random']['correct'] += 1

        # Each metric
        for metric in metrics:
            results[metric]['total'] += 1
            best = max(compiling, key=lambda c: c.metrics.get(metric, 0))
            if best.passes_tests:
                results[metric]['correct'] += 1

        # Ensemble: text-based
        results['ensemble_text']['total'] += 1
        def text_ensemble(c):
            return (c.metrics.get('bigram_sim', 0) * 0.4 +
                    c.metrics.get('trigram_sim', 0) * 0.4 +
                    c.metrics.get('token_f1', 0) * 0.2)
        if max(compiling, key=text_ensemble).passes_tests:
            results['ensemble_text']['correct'] += 1

        # Ensemble: finetuned UniXcoder + text
        results['ensemble_unixcoder_ft']['total'] += 1
        def unixcoder_ensemble(c):
            return (c.metrics.get('unixcoder_finetuned', 0) * 0.5 +
                    c.metrics.get('bigram_sim', 0) * 0.25 +
                    c.metrics.get('trigram_sim', 0) * 0.25)
        if max(compiling, key=unixcoder_ensemble).passes_tests:
            results['ensemble_unixcoder_ft']['correct'] += 1

        # Ensemble: perplexity + text
        results['ensemble_perplexity']['total'] += 1
        def perplexity_ensemble(c):
            return (c.metrics.get('neg_perplexity', float('-inf')) * 0.5 +
                    c.metrics.get('bigram_sim', 0) * 0.25 +
                    c.metrics.get('trigram_sim', 0) * 0.25)
        if max(compiling, key=perplexity_ensemble).passes_tests:
            results['ensemble_perplexity']['correct'] += 1

    # Report results
    print("\n" + "=" * 70)
    print("RESULTS: PASS@1 BY SELECTION METRIC")
    print("=" * 70)

    sorted_metrics = sorted(results.items(), key=lambda x: x[1]['correct']/max(x[1]['total'],1), reverse=True)

    for metric, stats in sorted_metrics:
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            print(f"  {metric:25s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:5.1f}%")

    # Save results
    results_file = cache_dir / f"selection_results_all{seed_suffix(config.seed)}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Detailed comparison: base vs finetuned UniXcoder
    print("\n" + "=" * 70)
    print("UNIXCODER COMPARISON: BASE vs FINETUNED")
    print("=" * 70)

    base_correct = results['unixcoder_base']['correct']
    ft_correct = results['unixcoder_finetuned']['correct']
    total = results['unixcoder_base']['total']

    print(f"\n  Base UniXcoder:      {base_correct}/{total} = {100*base_correct/total:.1f}%")
    print(f"  Finetuned UniXcoder: {ft_correct}/{total} = {100*ft_correct/total:.1f}%")
    print(f"  Improvement:         {ft_correct - base_correct:+d} samples")


if __name__ == "__main__":
    main()
