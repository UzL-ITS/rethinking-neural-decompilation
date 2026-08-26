#!/usr/bin/env python3
"""
Comprehensive evaluation of LLM4Decompile on obfuscated samples with selection.

This script:
1. Generates 100 candidates per sample using vLLM with temperature sampling
2. Filters by compilation
3. Evaluates perplexity, MBR-consensus, and execution-guided selection
4. Evaluates both original and fine-tuned models
5. Breaks down results by obfuscation type and optimization level
6. Saves detailed JSON results for LaTeX export

Metrics:
- Pass@1 (functional correctness)
- Compile rate
- Levenshtein similarity
- CodeBLEU score
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import subprocess
import tempfile
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
from pathlib import Path
from tqdm import tqdm
import pickle
import time
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

import Levenshtein
from codebleu import calc_codebleu
import shutil
import shlex
from concurrent.futures import ThreadPoolExecutor
import sys
import re
from collections import Counter
PROJECT_DIR = Path(__file__).resolve().parent
_SCRIPT_DIR = Path(__file__).resolve().parent

try:
    import tree_sitter_c as tsc
    from tree_sitter import Language, Parser

    C_LANGUAGE = Language(tsc.language())
    PARSER = Parser(C_LANGUAGE)
except Exception:
    C_LANGUAGE = None
    PARSER = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # Data
    obfuscated_data: str = str(_SCRIPT_DIR / "../RQ3_obfuscation_finetuning/train-test-split_v2/test_non_empty_functions.json")
    selected_samples_file: str = str(_SCRIPT_DIR / "cache_oracle/oracle_results.json")

    # Models
    original_model: str = "LLM4Binary/llm4decompile-9b-v2"
    finetuned_model: str = str(_SCRIPT_DIR / "checkpoints/llm4decompile-9b-obf-epoch1")

    # Generation settings
    num_candidates: int = 100
    temperatures: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9, 1.0])
    max_new_tokens: int = 4096
    max_sequence_len: int = 4096

    # Evaluation
    num_workers: int = 20
    execution_guided_checks: int = 3
    selection_methods: List[str] = field(default_factory=lambda: [
        "perplexity",
        "mbr_consensus",
        "execution_guided",
    ])

    # Output
    output_dir: str = str(_SCRIPT_DIR / "results_selection")
    cache_dir: str = str(_SCRIPT_DIR / "cache_selection")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Candidate:
    """A generated decompilation candidate."""
    code: str
    temperature: float
    compiles: bool = False
    passes_tests: bool = False
    perplexity: float = float('inf')
    mbr_consensus: float = 0.0
    execution_guided: float = 0.0
    levenshtein: float = 0.0
    codebleu: float = 0.0


@dataclass
class Sample:
    """A test sample with multiple candidates."""
    idx: int
    obf_pass: str
    opt_level: str
    ghidra_code: str
    c_func: str
    c_test: str
    real_io_pairs: dict
    real_iospec: str
    candidates: List[Candidate] = field(default_factory=list)

    selected_compiles: bool = False
    selected_passes_tests: bool = False
    selected_levenshtein: float = 0.0
    selected_codebleu: float = 0.0
    selection_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# =============================================================================
# Data Loading
# =============================================================================

def compute_total_sequence_length(ghidra_code: str, source_code: str, tokenizer) -> int:
    """Compute full prompt+target sequence length before truncation."""
    prompt = f"# This is the assembly code:\n{ghidra_code}\n# What is the source code?\n"
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


def compute_prompt_length(ghidra_code: str, tokenizer) -> int:
    prompt = build_prompt(ghidra_code)
    return len(
        tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
            return_tensors=None,
        )["input_ids"]
    )

def load_obfuscated_data(config: Config) -> List[Sample]:
    """Load obfuscated test data."""
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

        # Save sampled task IDs to list to check the samples selected later.
        sampled_ids = df["task_id"].tolist()
        with open("sampled_ids_1000.txt", "w") as f:
            for x in sampled_ids:
                f.write(f"{x}\n")
            
    samples = []
    if isinstance(raw_data, list):
        num_samples = len(raw_data)
        print(f"  Found {num_samples} samples")

        for idx, row in enumerate(raw_data):
            sample = Sample(
                idx=idx,
                obf_pass=str(row['obf_pass']),
                opt_level=row.get('type', row.get('opt_level')),
                ghidra_code=row.get('input_asm_prompt', row.get('c_code', row.get('c_code_obf'))),
                c_func=row['c_func'],
                c_test=row['c_test'],
                real_io_pairs=row['real_io_pairs'],
                real_iospec=row['real_iospec']
            )
            samples.append(sample)
    elif isinstance(raw_data, dict):
        num_samples = len(raw_data['func_name'])
        print(f"  Found {num_samples} samples")

        for idx in range(num_samples):
            str_idx = str(idx)
            code_column = raw_data.get('input_asm_prompt', raw_data.get('c_code', raw_data.get('c_code_obf')))
            sample = Sample(
                idx=idx,
                obf_pass=str(raw_data['obf_pass'][str_idx]),
                opt_level=raw_data['type'][str_idx],
                ghidra_code=code_column[str_idx],
                c_func=raw_data['c_func'][str_idx],
                c_test=raw_data['c_test'][str_idx],
            )
            samples.append(sample)
    else:
        raise TypeError(f"Unsupported JSON structure: {type(raw_data).__name__}")

    print("\n  Obfuscation pass distribution:")
    obf_counts = {}
    for s in samples:
        obf_counts[s.obf_pass] = obf_counts.get(s.obf_pass, 0) + 1
    for k, v in sorted(obf_counts.items()):
        print(f"    {k}: {v}")

    print("\n  Optimization level distribution:")
    opt_counts = {}
    for s in samples:
        opt_counts[s.opt_level] = opt_counts.get(s.opt_level, 0) + 1
    for k, v in sorted(opt_counts.items()):
        print(f"    {k}: {v}")

    return samples

# =============================================================================
# Prompt and Code Utilities
# =============================================================================

def build_prompt(ghidra_code: str) -> str:
    """Build LLM4Decompile prompt."""
    return f"# This is the assembly code:\n{ghidra_code}\n# What is the source code?\n"


def extract_c_code(text: str) -> str:
    """Extract C code from model output."""
    if "```c" in text:
        start = text.find("```c") + 4
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    return text.strip()


def extract_function_body(code: str) -> str:
    """Extract function body (remove includes)."""
    lines = [l for l in code.split('\n') if not l.strip().startswith('#include')]
    return '\n'.join(lines).strip()


def normalize_code_for_mbr(code: str) -> str:
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'\s+', ' ', code)
    return code.strip()


def tokenize_c_code(code: str) -> List[str]:
    pattern = r'[a-zA-Z_][a-zA-Z0-9_]*|0x[0-9a-fA-F]+|[0-9]+\.?[0-9]*|[+\-*/%=<>!&|^~?:;,.\[\]{}()]|"[^"]*"|\'[^\']*\''
    return re.findall(pattern, code)


def get_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def parse_code(code: str) -> Optional[Any]:
    if PARSER is None:
        return None
    try:
        tree = PARSER.parse(bytes(code, "utf8"))
        return tree.root_node
    except Exception:
        return None


def get_node_type_counts(node: Any) -> Counter:
    counts = Counter()

    def traverse(n: Any) -> None:
        counts[n.type] += 1
        for child in n.children:
            traverse(child)

    if node:
        traverse(node)
    return counts


def get_control_flow_counts(node: Any) -> Dict[str, int]:
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

    def traverse(n: Any) -> None:
        if n.type in counts:
            counts[n.type] += 1
        for child in n.children:
            traverse(child)

    if node:
        traverse(node)
    return counts


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


def derive_execution_guided_prefixes(real_io_pairs: dict, max_checks: int = 3) -> List[dict]:
    inputs = real_io_pairs.get('input', [])
    outputs = real_io_pairs.get('output', [])
    num_cases = min(len(inputs), len(outputs))

    if num_cases == 0:
        return []
    if num_cases == 1:
        keep_counts = [1]
    else:
        keep_counts = []
        for value in [1, max(1, num_cases // 2), num_cases - 1]:
            if 1 <= value <= num_cases and value not in keep_counts:
                keep_counts.append(value)

    prefixes = []
    for keep_count in keep_counts[:max_checks]:
        prefixes.append({
            'input': inputs[:keep_count],
            'output': outputs[:keep_count],
        })
    return prefixes


def compute_execution_guided_score(
    code: str,
    c_func: str,
    c_test: str,
    real_io_pairs: dict,
    real_iospec: str,
    max_checks: int = 3,
) -> float:
    checkpoints = derive_execution_guided_prefixes(real_io_pairs, max_checks)
    if not checkpoints:
        return 0.0

    passed = 0
    cfg = Config()
    for partial_real_io_pairs in checkpoints:
        if run_test((code, c_func, c_test, partial_real_io_pairs, real_iospec), cfg):
            passed += 1
    return passed / len(checkpoints)


def compute_execution_guided_score_from_args(args: Tuple[str, str, str, dict, str, int]) -> float:
    return compute_execution_guided_score(*args)


# =============================================================================
# Compilation and Testing
# =============================================================================

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
                    timeout=20,
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
                print(e)
                all_match = False
                break

        return all_match
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Metrics
# =============================================================================

def compute_levenshtein(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    dist = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (dist / max_len)


def compute_codebleu(reference: str, candidate: str) -> float:
    """CodeBLEU score."""
    try:
        if not reference.strip() or not candidate.strip():
            return 0.0
        result = calc_codebleu(
            references=[[reference]],
            predictions=[candidate],
            lang="c",
            weights=(0.25, 0.25, 0.25, 0.25),
            tokenizer=None
        )
        return result['codebleu']
    except:
        return 0.0


# =============================================================================
# Perplexity Computation
# =============================================================================

class PerplexityComputer:
    """Compute perplexity for candidate selection."""

    def __init__(self, model_path: str, max_sequence_len: int = 4096):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_sequence_len = max_sequence_len
        print(f"Loading model for perplexity: {model_path}")

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

    def compute(self, ghidra_code: str, decompiled_code: str) -> float:
        """Compute perplexity of decompiled code given Ghidra prompt."""
        try:
            prompt = build_prompt(ghidra_code)
            full_text = prompt + decompiled_code

            prompt_tokens = self.tokenizer(prompt, return_tensors="pt", truncation=False)
            full_tokens = self.tokenizer(full_text, return_tensors="pt", truncation=False)

            prompt_len = prompt_tokens.input_ids.shape[1]
            full_len = full_tokens.input_ids.shape[1]
            if full_len > self.max_sequence_len:
                return float('inf')
            decompiled_len = full_len - prompt_len

            if decompiled_len < 5:
                return float('inf')

            input_ids = full_tokens.input_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits

            target_logits = logits[0, prompt_len-1:full_len-1, :]
            target_tokens = input_ids[0, prompt_len:full_len]

            log_probs = F.log_softmax(target_logits, dim=-1)
            token_log_probs = log_probs.gather(1, target_tokens.unsqueeze(1)).squeeze(1)

            avg_log_prob = token_log_probs.float().mean().item()
            perplexity = float(torch.exp(torch.tensor(-avg_log_prob)).item())

            return perplexity
        except Exception as e:
            return float('inf')


# =============================================================================
# Candidate Generation with vLLM
# =============================================================================

def generate_candidates(
    model_path: str,
    samples: List[Sample],
    config: Config,
    cache_file: Path
) -> List[Sample]:
    """Generate candidates using vLLM."""

    incremental_cache = cache_file.with_name(cache_file.stem + "_incremental.pkl")

    if cache_file.exists():
        print(f"\nLoading cached candidates from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"\n=== Generating Candidates with vLLM ===")
    print(f"Model: {model_path}")
    print(f"Candidates per sample: {config.num_candidates}")
    print(f"Temperatures: {config.temperatures}")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype='bfloat16',
        max_model_len=config.max_sequence_len,
        gpu_memory_utilization=0.75,
        tensor_parallel_size=1,
    )

    tokenizer = llm.get_tokenizer()
    valid_samples = []
    skipped = 0
    prompt_lengths = {}
    available_tokens = {}
    for s in samples:
        token_len = compute_prompt_length(s.ghidra_code, tokenizer)
        prompt_lengths[s.idx] = token_len
        if token_len < config.max_sequence_len:
            valid_samples.append(s)
            available_tokens[s.idx] = config.max_sequence_len - token_len
        else:
            skipped += 1

    print(f"Valid samples: {len(valid_samples)}, Skipped (too long): {skipped}")

    candidates_per_temp = config.num_candidates // len(config.temperatures)
    sample_to_candidates = {s.idx: [] for s in valid_samples}

    completed_temp_indices = set()
    if incremental_cache.exists():
        print(f"\nLoading incremental candidate cache from {incremental_cache}")
        with open(incremental_cache, 'rb') as f:
            incremental_data = pickle.load(f)
        cached_candidates = incremental_data.get('sample_to_candidates', {})
        for key, value in cached_candidates.items():
            sample_to_candidates[key] = value
        completed_temp_indices = set(incremental_data.get('completed_temp_indices', []))
        print(f"  Completed temperatures: {sorted(completed_temp_indices)}")

    def save_incremental() -> None:
        with open(incremental_cache, 'wb') as f:
            pickle.dump(
                {
                    'sample_to_candidates': sample_to_candidates,
                    'completed_temp_indices': sorted(completed_temp_indices),
                },
                f,
            )

    def iter_budget_batches(sample_group: List[Sample], batch_size: int = 32):
        sorted_samples = sorted(sample_group, key=lambda s: available_tokens[s.idx], reverse=True)
        for start in range(0, len(sorted_samples), batch_size):
            batch = sorted_samples[start:start + batch_size]
            batch_budget = min(available_tokens[s.idx] for s in batch)
            yield batch, batch_budget

    for temp_idx, temp in enumerate(tqdm(config.temperatures, desc="Temperatures")):
        if temp_idx in completed_temp_indices:
            continue

        for batch_samples, batch_budget in tqdm(
            list(iter_budget_batches(valid_samples, batch_size=32)),
            desc=f"Temp {temp}",
            leave=False
        ):
            if batch_budget <= 0:
                continue

            params = SamplingParams(
                temperature=max(temp, 0.01),
                max_tokens=min(config.max_new_tokens, batch_budget),
                top_p=0.95 if temp > 0 else 1.0,
                n=candidates_per_temp,
            )

            prompts = [build_prompt(sample.ghidra_code) for sample in batch_samples]
            outputs = llm.generate(prompts, params)

            for sample, output in zip(batch_samples, outputs):
                for out in output.outputs:
                    code = extract_c_code(out.text)
                    sample_to_candidates[sample.idx].append(
                        Candidate(code=code, temperature=temp)
                    )

        completed_temp_indices.add(temp_idx)
        save_incremental()

    for sample in valid_samples:
        sample.candidates = sample_to_candidates[sample.idx]

    del llm
    torch.cuda.empty_cache()

    print(f"Saving candidates to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(samples, f)

    if incremental_cache.exists():
        incremental_cache.unlink()

    return samples


# =============================================================================
# Full Evaluation Pipeline
# =============================================================================

def ensure_runtime_fields(samples: List[Sample]) -> None:
    """Populate fields that may be missing when loading older pickle caches."""
    for sample in samples:
        if not hasattr(sample, "selection_results"):
            sample.selection_results = {}
        for candidate in sample.candidates:
            if not hasattr(candidate, "mbr_consensus"):
                candidate.mbr_consensus = 0.0
            if not hasattr(candidate, "execution_guided"):
                candidate.execution_guided = 0.0
            if not hasattr(candidate, "perplexity"):
                candidate.perplexity = float("inf")


def select_candidate(compiling: List[Candidate], method: str) -> Optional[Candidate]:
    if not compiling:
        return None
    if method == "perplexity":
        return min(compiling, key=lambda c: (c.perplexity, -c.mbr_consensus, -c.execution_guided))
    if method == "mbr_consensus":
        return max(compiling, key=lambda c: (c.mbr_consensus, -c.perplexity, c.execution_guided))
    if method == "execution_guided":
        return max(compiling, key=lambda c: (c.execution_guided, -c.perplexity, c.mbr_consensus))
    raise ValueError(f"Unknown selection method: {method}")


def record_selected_candidate(sample: Sample, method: str, candidate: Candidate) -> None:
    orig_clean = extract_function_body(sample.c_func)
    gen_clean = extract_function_body(candidate.code)
    levenshtein = compute_levenshtein(orig_clean, gen_clean)
    codebleu = compute_codebleu(orig_clean, gen_clean)
    sample.selection_results[method] = {
        "candidate_code": candidate.code,
        "compiles": True,
        "passes_tests": False,
        "levenshtein": levenshtein,
        "codebleu": codebleu,
        "perplexity": candidate.perplexity,
        "mbr_consensus": candidate.mbr_consensus,
        "execution_guided": candidate.execution_guided,
    }


def compute_or_load_mbr_scores(samples: List[Sample], cache_dir: Path, model_key: str) -> None:
    mbr_cache = cache_dir / f"mbr_consensus_{model_key}.pkl"
    if mbr_cache.exists():
        print("\nLoading cached MBR consensus results...")
        with open(mbr_cache, "rb") as f:
            mbr_results = pickle.load(f)
    else:
        mbr_results = {}

    dirty = False
    print("\n=== Computing MBR Consensus ===")
    for s in tqdm(samples, desc="MBR"):
        compiling = [(i, c) for i, c in enumerate(s.candidates) if c.compiles]
        if not compiling:
            continue

        missing = [(i, c) for i, c in compiling if (s.idx, i) not in mbr_results]
        if missing:
            scores = compute_mbr_scores([c for _, c in compiling])
            for (cand_idx, _), score in zip(compiling, scores):
                mbr_results[(s.idx, cand_idx)] = score
            dirty = True

        for cand_idx, candidate in compiling:
            candidate.mbr_consensus = mbr_results.get((s.idx, cand_idx), 0.0)

    if dirty or not mbr_cache.exists():
        with open(mbr_cache, "wb") as f:
            pickle.dump(mbr_results, f)


def compute_or_load_execution_guided_scores(
    samples: List[Sample],
    cache_dir: Path,
    model_key: str,
    config: Config,
) -> None:
    eg_cache = cache_dir / f"execution_guided_{model_key}.pkl"
    if eg_cache.exists():
        print("\nLoading cached execution-guided results...")
        with open(eg_cache, "rb") as f:
            eg_results = pickle.load(f)
    else:
        eg_results = {}

    dirty = False
    print("\n=== Computing Execution-Guided Scores ===")
    for s in tqdm(samples, desc="Execution-guided"):
        compiling = [(i, c) for i, c in enumerate(s.candidates) if c.compiles]
        if not compiling:
            continue

        missing = [(i, c) for i, c in compiling if (s.idx, i) not in eg_results]
        if missing:
            tasks = [
                (c.code, s.c_func, s.c_test, s.real_io_pairs, s.real_iospec, config.execution_guided_checks)
                for _, c in missing
            ]
            with ThreadPoolExecutor(max_workers=max(1, min(len(tasks), config.num_workers))) as ex:
                scores = list(ex.map(compute_execution_guided_score_from_args, tasks))
            for (cand_idx, _), score in zip(missing, scores):
                eg_results[(s.idx, cand_idx)] = score
            dirty = True

        for cand_idx, candidate in compiling:
            candidate.execution_guided = eg_results.get((s.idx, cand_idx), 0.0)

    if dirty or not eg_cache.exists():
        with open(eg_cache, "wb") as f:
            pickle.dump(eg_results, f)


def compute_or_load_oracle_first20(samples: List[Sample], cache_dir: Path, model_key: str, config: Config) -> int:
    oracle_cache = cache_dir / f"oracle_first20_{model_key}.pkl"
    if oracle_cache.exists():
        print("\nLoading cached first-20 oracle results...")
        with open(oracle_cache, "rb") as f:
            oracle_results = pickle.load(f)
    else:
        oracle_results = {}

    dirty = False
    for s in tqdm(samples, desc="Oracle first20"):
        if s.idx in oracle_results:
            continue
        compiling = [c for c in s.candidates if c.compiles]
        passed = False
        if compiling:
            test_tasks = [(c.code, s.c_func, s.c_test, s.real_io_pairs, s.real_iospec) for c in compiling[:20]]
            for task in test_tasks:
                if run_test(task, config):
                    passed = True
                    break
        oracle_results[s.idx] = passed
        dirty = True

    if dirty or not oracle_cache.exists():
        with open(oracle_cache, "wb") as f:
            pickle.dump(oracle_results, f)

    return sum(1 for passed in oracle_results.values() if passed)


def evaluate_model(
    model_name: str,
    model_path: str,
    samples: List[Sample],
    config: Config,
) -> Dict:
    """Full evaluation pipeline for a single model."""

    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    model_key = model_name.replace(' ', '_').replace('/', '_')

    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")

    # Step 1: Generate candidates
    cand_cache = cache_dir / f"candidates_{model_key}.pkl"
    samples = generate_candidates(model_path, samples, config, cand_cache)
    ensure_runtime_fields(samples)

    # Step 2: Check compilation
    compile_cache = cache_dir / f"compilation_{model_key}.pkl"

    if compile_cache.exists():
        print("\nLoading cached compilation results...")
        with open(compile_cache, 'rb') as f:
            compile_results = pickle.load(f)
        for s in samples:
            for i, c in enumerate(s.candidates):
                key = (s.idx, i)
                if key in compile_results:
                    c.compiles = compile_results[key]
    else:
        print("\n=== Checking Compilation ===")
        tasks = []
        task_info = []
        for s in samples:
            for i, c in enumerate(s.candidates):
                tasks.append((c.code, s.c_func, s.c_test))
                task_info.append((s.idx, i))

        print(f"Checking {len(tasks)} candidates...")
        with mp.Pool(config.num_workers) as pool:
            results = list(tqdm(pool.imap(check_compilation, tasks), total=len(tasks)))

        compile_results = {}
        for (sample_idx, cand_idx), compiles in zip(task_info, results):
            compile_results[(sample_idx, cand_idx)] = compiles
            for s in samples:
                if s.idx == sample_idx:
                    s.candidates[cand_idx].compiles = compiles
                    break

        with open(compile_cache, 'wb') as f:
            pickle.dump(compile_results, f)

    total_compiling = sum(sum(1 for c in s.candidates if c.compiles) for s in samples)
    print(f"Total compiling candidates: {total_compiling}")

    # Step 3: Compute perplexity for compiling candidates
    ppl_cache = cache_dir / f"perplexity_{model_key}.pkl"

    if ppl_cache.exists():
        print("\nLoading cached perplexity results...")
        with open(ppl_cache, 'rb') as f:
            ppl_results = pickle.load(f)
        for s in samples:
            for i, c in enumerate(s.candidates):
                key = (s.idx, i)
                if key in ppl_results:
                    c.perplexity = ppl_results[key]
    else:
        print("\n=== Computing Perplexity ===")
        ppl_computer = PerplexityComputer(model_path, max_sequence_len=config.max_sequence_len)

        ppl_results = {}
        for s in tqdm(samples, desc="Perplexity"):
            for i, c in enumerate(s.candidates):
                if c.compiles:
                    ppl = ppl_computer.compute(s.ghidra_code, c.code)
                    c.perplexity = ppl
                    ppl_results[(s.idx, i)] = ppl

        del ppl_computer
        torch.cuda.empty_cache()

        with open(ppl_cache, 'wb') as f:
            pickle.dump(ppl_results, f)

    # Step 4: Compute additional selection scores
    compute_or_load_mbr_scores(samples, cache_dir, model_key)
    compute_or_load_execution_guided_scores(samples, cache_dir, model_key, config)

    # Step 5: Select best candidate per sample for each method
    print("\n=== Selecting Best Candidates ===")
    selection_methods = list(config.selection_methods)
    for s in samples:
        s.selection_results = {}
        compiling = [c for c in s.candidates if c.compiles]
        if not compiling:
            continue
        for method in selection_methods:
            best = select_candidate(compiling, method)
            if best is not None:
                record_selected_candidate(s, method, best)

    # Step 6: Run tests on selected candidates for each method
    for method in selection_methods:
        test_cache = cache_dir / f"tests_{model_key}_{method}.pkl"
        legacy_test_cache = cache_dir / f"tests_{model_key}.pkl"

        if test_cache.exists():
            print(f"\nLoading cached test results for {method}...")
            with open(test_cache, 'rb') as f:
                test_results = pickle.load(f)
        elif method == "perplexity" and legacy_test_cache.exists():
            print("\nLoading legacy cached test results for perplexity...")
            with open(legacy_test_cache, 'rb') as f:
                test_results = pickle.load(f)
        else:
            test_results = {}

        test_tasks = []
        test_indices = []
        for s in samples:
            selected = s.selection_results.get(method)
            if not selected or s.idx in test_results:
                continue
            test_tasks.append((selected["candidate_code"], s.c_func, s.c_test, s.real_io_pairs, s.real_iospec))
            test_indices.append(s.idx)

        if test_tasks:
            print(f"\n=== Running Tests for {method} ===")
            print(f"Testing {len(test_tasks)} selected candidates...")
            with ThreadPoolExecutor(max_workers=max(1, min(len(test_tasks), config.num_workers))) as ex:
                results = list(tqdm(ex.map(lambda t: run_test(t, config), test_tasks), total=len(test_tasks)))

            for idx, passes in zip(test_indices, results):
                test_results[idx] = passes

            with open(test_cache, 'wb') as f:
                pickle.dump(test_results, f)

        for s in samples:
            selected = s.selection_results.get(method)
            if selected and s.idx in test_results:
                selected["passes_tests"] = test_results[s.idx]

    # Preserve old top-level selected_* fields for the default perplexity method.
    for s in samples:
        selected = s.selection_results.get("perplexity")
        if selected:
            s.selected_compiles = bool(selected["compiles"])
            s.selected_passes_tests = bool(selected["passes_tests"])
            s.selected_levenshtein = float(selected["levenshtein"])
            s.selected_codebleu = float(selected["codebleu"])

    # Step 7: Compute statistics
    oracle_pass = compute_or_load_oracle_first20(samples, cache_dir, model_key, config)
    method_stats = {
        method: compute_statistics(samples, model_name, config, method, oracle_pass)
        for method in selection_methods
    }
    primary_method = "execution_guided" if "execution_guided" in method_stats else (
        "perplexity" if "perplexity" in method_stats else selection_methods[0]
    )
    stats = dict(method_stats[primary_method])
    stats["selection_methods"] = method_stats
    stats["primary_method"] = primary_method
    return stats


def compute_statistics(
    samples: List[Sample],
    model_name: str,
    config: Config,
    selection_method: str = "perplexity",
    oracle_pass: Optional[int] = None,
) -> Dict:
    """Compute comprehensive statistics."""

    n_total = len(samples)
    selected = [s.selection_results.get(selection_method, {}) for s in samples]
    n_compile = sum(1 for item in selected if item.get("compiles"))
    n_pass = sum(1 for item in selected if item.get("passes_tests"))

    stats = {
        'model': model_name,
        'config': {
            'num_candidates': config.num_candidates,
            'temperatures': config.temperatures,
            'selection_method': selection_method,
        },
        'total': n_total,
        'compile': n_compile,
        'compile_rate': n_compile / n_total * 100 if n_total > 0 else 0,
        'pass': n_pass,
        'pass_rate': n_pass / n_total * 100 if n_total > 0 else 0,
        'levenshtein': sum(float(item.get("levenshtein", 0.0)) for item in selected) / n_total * 100 if n_total > 0 else 0,
        'codebleu': sum(float(item.get("codebleu", 0.0)) for item in selected) / n_total * 100 if n_total > 0 else 0,
    }

    if oracle_pass is None:
        oracle_pass = 0
        for s in samples:
            compiling = [c for c in s.candidates if c.compiles]
            if compiling:
                test_tasks = [(c.code, s.c_func, s.c_test, s.real_io_pairs, s.real_iospec) for c in compiling[:20]]
                for task in test_tasks:
                    if run_test(task, config):
                        oracle_pass += 1
                        break
    stats['oracle_pass'] = oracle_pass
    stats['oracle_pass_rate'] = oracle_pass / n_total * 100 if n_total > 0 else 0

    # By obfuscation pass
    stats['by_obf_pass'] = {}
    obf_passes = set(s.obf_pass for s in samples)
    for obf in obf_passes:
        obf_samples = [s for s in samples if s.obf_pass == obf]
        n = len(obf_samples)
        obf_selected = [s.selection_results.get(selection_method, {}) for s in obf_samples]
        stats['by_obf_pass'][obf] = {
            'total': n,
            'pass_rate': sum(1 for item in obf_selected if item.get("passes_tests")) / n * 100 if n > 0 else 0,
            'compile_rate': sum(1 for item in obf_selected if item.get("compiles")) / n * 100 if n > 0 else 0,
            'levenshtein': sum(float(item.get("levenshtein", 0.0)) for item in obf_selected) / n * 100 if n > 0 else 0,
            'codebleu': sum(float(item.get("codebleu", 0.0)) for item in obf_selected) / n * 100 if n > 0 else 0,
        }

    # By optimization level
    stats['by_opt_level'] = {}
    opt_levels = set(s.opt_level for s in samples)
    for opt in opt_levels:
        opt_samples = [s for s in samples if s.opt_level == opt]
        n = len(opt_samples)
        opt_selected = [s.selection_results.get(selection_method, {}) for s in opt_samples]
        stats['by_opt_level'][opt] = {
            'total': n,
            'pass_rate': sum(1 for item in opt_selected if item.get("passes_tests")) / n * 100 if n > 0 else 0,
            'compile_rate': sum(1 for item in opt_selected if item.get("compiles")) / n * 100 if n > 0 else 0,
            'levenshtein': sum(float(item.get("levenshtein", 0.0)) for item in opt_selected) / n * 100 if n > 0 else 0,
            'codebleu': sum(float(item.get("codebleu", 0.0)) for item in opt_selected) / n * 100 if n > 0 else 0,
        }

    return stats


def print_statistics(stats: Dict):
    """Print formatted statistics."""
    print(f"\n{'='*70}")
    print(f"RESULTS: {stats['model']}")
    print(f"{'='*70}")

    print(f"\nSelection: {stats['config']['num_candidates']} candidates, {stats['config']['selection_method']}")

    print(f"\nOverall:")
    print(f"  Samples:      {stats['total']:,}")
    print(f"  Compile rate: {stats['compile']:,}/{stats['total']:,} = {stats['compile_rate']:.1f}%")
    print(f"  Pass@1:       {stats['pass']:,}/{stats['total']:,} = {stats['pass_rate']:.1f}%")
    print(f"  Levenshtein:  {stats['levenshtein']:.1f}%")
    print(f"  CodeBLEU:     {stats['codebleu']:.1f}%")
    print(f"  Oracle@20:    {stats['oracle_pass']:,}/{stats['total']:,} = {stats['oracle_pass_rate']:.1f}%")

    if "selection_methods" in stats:
        print(f"\nSelection Methods:")
        print(f"  {'Method':<20} {'Pass@1':>8} {'Compile':>8} {'Lev':>8} {'CodeBLEU':>8}")
        print(f"  {'-'*58}")
        for method, method_stats in stats["selection_methods"].items():
            print(
                f"  {method:<20} {method_stats['pass_rate']:>7.1f}% "
                f"{method_stats['compile_rate']:>7.1f}% "
                f"{method_stats['levenshtein']:>7.1f}% "
                f"{method_stats['codebleu']:>7.1f}%"
            )

    print(f"\nBy Obfuscation Pass:")
    print(f"  {'Pass':<20} {'Pass@1':>8} {'Compile':>8} {'Lev':>8} {'CodeBLEU':>8} {'n':>6}")
    print(f"  {'-'*60}")
    for obf, data in sorted(stats['by_obf_pass'].items()):
        print(f"  {obf:<20} {data['pass_rate']:>7.1f}% {data['compile_rate']:>7.1f}% "
              f"{data['levenshtein']:>7.1f}% {data['codebleu']:>7.1f}% {data['total']:>6}")

    print(f"\nBy Optimization Level:")
    print(f"  {'Opt':<6} {'Pass@1':>8} {'Compile':>8} {'Lev':>8} {'CodeBLEU':>8} {'n':>6}")
    print(f"  {'-'*50}")
    for opt, data in sorted(stats['by_opt_level'].items()):
        print(f"  {opt:<6} {data['pass_rate']:>7.1f}% {data['compile_rate']:>7.1f}% "
              f"{data['levenshtein']:>7.1f}% {data['codebleu']:>7.1f}% {data['total']:>6}")


def print_comparison(stats_orig: Dict, stats_ft: Dict):
    """Print comparison between models."""
    print(f"\n{'='*70}")
    print("COMPARISON: ORIGINAL vs FINE-TUNED (with selection)")
    print(f"{'='*70}")

    print(f"\n{'Metric':<20} {'Original':>12} {'Fine-tuned':>12} {'Delta':>12}")
    print("-" * 58)

    for name, key in [('Pass@1 (%)', 'pass_rate'), ('Compile (%)', 'compile_rate'),
                      ('Levenshtein (%)', 'levenshtein'), ('CodeBLEU (%)', 'codebleu')]:
        orig = stats_orig[key]
        ft = stats_ft[key]
        delta = ft - orig
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        print(f"{name:<20} {orig:>11.1f}% {ft:>11.1f}% {delta_str:>11}pp")

    if "selection_methods" in stats_orig and "selection_methods" in stats_ft:
        print(f"\nBy Selection Method (Pass@1):")
        print(f"  {'Method':<20} {'Original':>10} {'Fine-tuned':>12} {'Delta':>10}")
        print(f"  {'-'*56}")
        for method in stats_orig["selection_methods"]:
            if method not in stats_ft["selection_methods"]:
                continue
            orig = stats_orig["selection_methods"][method]["pass_rate"]
            ft = stats_ft["selection_methods"][method]["pass_rate"]
            delta = ft - orig
            delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
            print(f"  {method:<20} {orig:>9.1f}% {ft:>11.1f}% {delta_str:>9}pp")

    print(f"\nBy Obfuscation Pass (Pass@1):")
    print(f"  {'Obf Pass':<20} {'Original':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print(f"  {'-'*55}")
    for obf in sorted(stats_orig['by_obf_pass'].keys()):
        orig = stats_orig['by_obf_pass'][obf]['pass_rate']
        ft = stats_ft['by_obf_pass'][obf]['pass_rate']
        delta = ft - orig
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        print(f"  {obf:<20} {orig:>9.1f}% {ft:>11.1f}% {delta_str:>9}pp")

    if "by_opt_level" in stats_orig and "by_opt_level" in stats_ft:
        print(f"\nBy Optimization Level (Pass@1):")
        print(f"  {'Opt Level':<20} {'Original':>10} {'Fine-tuned':>12} {'Delta':>10} {'n':>6}")
        print(f"  {'-'*63}")
        for opt in sorted(stats_orig['by_opt_level'].keys()):
            if opt not in stats_ft['by_opt_level']:
                continue
            orig_data = stats_orig['by_opt_level'][opt]
            ft_data = stats_ft['by_opt_level'][opt]
            orig = orig_data['pass_rate']
            ft = ft_data['pass_rate']
            delta = ft - orig
            delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
            n = orig_data.get('total', ft_data.get('total', 0))
            print(f"  {opt:<20} {orig:>9.1f}% {ft:>11.1f}% {delta_str:>9}pp {n:>6}")


# =============================================================================
# Main
# =============================================================================

def main():
    config = Config()

    print("=" * 70)
    print("OBFUSCATED DECOMPILATION EVALUATION WITH SELECTION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Candidates per sample: {config.num_candidates}")
    print(f"  Temperatures: {config.temperatures}")
    print(f"  Selection methods: {config.selection_methods}")
    print(f"  Original model: {config.original_model}")
    print(f"  Fine-tuned model: {config.finetuned_model}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    samples = load_obfuscated_data(config)

    # Evaluate original model
    samples_orig = [Sample(
        idx=s.idx, obf_pass=s.obf_pass, opt_level=s.opt_level,
        ghidra_code=s.ghidra_code, c_func=s.c_func, c_test=s.c_test,real_io_pairs=s.real_io_pairs, real_iospec=s.real_iospec) for s in samples]

    stats_original = evaluate_model(
        "Original LLM4Decompile-9B",
        config.original_model,
        samples_orig,
        config
    )
    print_statistics(stats_original)

    # Evaluate fine-tuned model
    samples_ft = [Sample(
        idx=s.idx, obf_pass=s.obf_pass, opt_level=s.opt_level,
        ghidra_code=s.ghidra_code, c_func=s.c_func, c_test=s.c_test, real_io_pairs=s.real_io_pairs, real_iospec=s.real_iospec
    ) for s in samples]

    stats_finetuned = evaluate_model(
        "Fine-tuned LLM4Decompile-9B",
        config.finetuned_model,
        samples_ft,
        config
    )
    print_statistics(stats_finetuned)

    print_comparison(stats_original, stats_finetuned)

    # Save results
    summary = {
        'config': {
            'num_candidates': config.num_candidates,
            'temperatures': config.temperatures,
            'selection_methods': config.selection_methods,
            'original_model': config.original_model,
            'finetuned_model': config.finetuned_model,
            'num_samples': len(samples),
        },
        'original': stats_original,
        'finetuned': stats_finetuned,
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Save LaTeX-friendly format
    latex_data = {
        'overall': {
            'metrics': ['Pass@1', 'Compile', 'Levenshtein', 'CodeBLEU'],
            'original': [
                stats_original['pass_rate'],
                stats_original['compile_rate'],
                stats_original['levenshtein'],
                stats_original['codebleu'],
            ],
            'finetuned': [
                stats_finetuned['pass_rate'],
                stats_finetuned['compile_rate'],
                stats_finetuned['levenshtein'],
                stats_finetuned['codebleu'],
            ],
        },
        'by_obf_pass': {},
        'by_opt_level': {},
        'selection_methods': {
            'original': {
                method: {
                    'pass_rate': stats['pass_rate'],
                    'compile_rate': stats['compile_rate'],
                    'levenshtein': stats['levenshtein'],
                    'codebleu': stats['codebleu'],
                }
                for method, stats in stats_original.get('selection_methods', {}).items()
            },
            'finetuned': {
                method: {
                    'pass_rate': stats['pass_rate'],
                    'compile_rate': stats['compile_rate'],
                    'levenshtein': stats['levenshtein'],
                    'codebleu': stats['codebleu'],
                }
                for method, stats in stats_finetuned.get('selection_methods', {}).items()
            },
        },
    }

    for obf in sorted(stats_original['by_obf_pass'].keys()):
        latex_data['by_obf_pass'][obf] = {
            'original': stats_original['by_obf_pass'][obf],
            'finetuned': stats_finetuned['by_obf_pass'][obf],
        }

    for opt in sorted(stats_original['by_opt_level'].keys()):
        latex_data['by_opt_level'][opt] = {
            'original': stats_original['by_opt_level'][opt],
            'finetuned': stats_finetuned['by_opt_level'][opt],
        }

    with open(output_dir / "latex_data.json", 'w') as f:
        json.dump(latex_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to {output_dir}/")
    print(f"  - summary.json: Full results")
    print(f"  - latex_data.json: LaTeX-friendly format")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
