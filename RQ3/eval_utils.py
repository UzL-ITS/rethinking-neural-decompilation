#!/usr/bin/env python3
"""
Shared utility functions for decompilation evaluation.

This module provides common functionality for:
- Code compilation and testing
- Metric computation (Levenshtein, CodeBLEU)
- LLM4Decompile prompt formatting
"""

import os
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import logging

import Levenshtein
from codebleu import calc_codebleu

_SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# Prompt formatting
# =============================================================================
class CodeBleuWarningCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())
        
def build_llm4decompile_prompt(ghidra_code: str) -> str:
    """Build the standard LLM4Decompile prompt."""
    return f"""# This is the assembly code:
{ghidra_code}
# What is the source code?
"""


def extract_c_code(text: str) -> str:
    """Extract C code from model output (handles markdown blocks)."""
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
    """Extract just the function body for fairer comparison (removes includes)."""
    lines = []
    for line in code.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#include'):
            continue
        lines.append(line)
    return '\n'.join(lines).strip()


# =============================================================================
# Compilation and Testing
# =============================================================================

def simplify_type(type_str: str) -> str:
    """Reduce complex C types to basic JSON-friendly types."""
    lowered = type_str.lower()
    if 'string' in lowered:
        return 'string'
    if 'bool' in lowered:
        return 'bool'
    if 'float' in lowered:
        return 'float'
    if 'int' in lowered:
        return 'int'
    return 'string'


def check_compilation(code: str, c_func: str, c_test: str, timeout: float = 10.0) -> bool:
    """Compile generated code using the validator-style C++ harness."""
    curr_dir = str(_SCRIPT_DIR)
    pattern = r'extern\s+"C"\s*\{[\s\S]*?#include\s+["<][^">]+[">][\s\S]*?\}'
    try:
        final_content = re.sub(pattern, lambda _: code.strip(), c_test)
    except Exception:
        return False

    temp_dir = tempfile.mkdtemp(prefix="compile_", dir=curr_dir)
    try:
        c_file = os.path.join(temp_dir, 'test.cpp')
        exe = os.path.join(temp_dir, 'test')
        with open(c_file, 'w', encoding='utf-8') as f:
            f.write(final_content)

        try:
            result = subprocess.run(
                f'g++ {c_file} -I . -o {exe}',
                shell=True,
                capture_output=True,
                timeout=timeout,
            )
            return result.returncode == 0
        except Exception:
            return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_functional_test(
    code: str,
    c_func: str,
    c_test: str,
    real_io_pairs: dict,
    real_iospec: str,
    timeout: float = 20.0,
) -> bool:
    """Run the validator-style functional test used by the oracle pipeline."""
    typemap = json.loads(real_iospec)['typemap']
    inputs_data = real_io_pairs['input']
    expected_outputs = real_io_pairs['output']
    curr_dir = str(_SCRIPT_DIR)
    pattern = r'extern\s+"C"\s*\{[\s\S]*?#include\s+["<][^">]+[">][\s\S]*?\}'

    try:
        final_content = re.sub(pattern, lambda _: code.strip(), c_test)
    except Exception:
        return False

    temp_dir = tempfile.mkdtemp(prefix="runtest_", dir=curr_dir)
    try:
        c_file = os.path.join(temp_dir, 'test.cpp')
        exe = os.path.join(temp_dir, 'test')
        input_json = os.path.join(temp_dir, 'input_tmp.json')
        output_json = os.path.join(temp_dir, 'output_tmp.json')

        with open(c_file, 'w', encoding='utf-8') as f:
            f.write(final_content)

        try:
            result = subprocess.run(
                f'g++ -O0 {c_file} -I . -o {exe}',
                shell=True,
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False
        except Exception:
            return False

        for inp_dict, exp_out in zip(inputs_data, expected_outputs):
            test_case_input = {}
            for var_name, raw_val in zip(inp_dict['var'], inp_dict['value']):
                base_type = simplify_type(typemap.get(var_name, "string"))
                try:
                    parsed_val = json.loads(raw_val)
                except Exception:
                    parsed_val = raw_val

                if base_type == "int":
                    test_case_input[var_name] = [int(x) for x in parsed_val] if isinstance(parsed_val, list) else int(parsed_val)
                elif base_type == "float":
                    test_case_input[var_name] = [float(x) for x in parsed_val] if isinstance(parsed_val, list) else float(parsed_val)
                elif base_type == "bool":
                    test_case_input[var_name] = bool(parsed_val)
                else:
                    test_case_input[var_name] = str(parsed_val)

            with open(input_json, "w", encoding='utf-8') as f:
                json.dump(test_case_input, f, indent=4)

            try:
                result = subprocess.run(
                    f'{os.path.join(curr_dir, "run-sandboxed.sh")} ./test input_tmp.json output_tmp.json',
                    cwd=temp_dir,
                    shell=True,
                    capture_output=True,
                    timeout=timeout,
                )
                if result.returncode != 0:
                    return False

                with open(output_json, "r", encoding='utf-8') as f:
                    actual_output_json = json.load(f)

                for var_name, raw_expected in zip(exp_out['var'], exp_out['value']):
                    expected_val = json.loads(raw_expected)
                    if actual_output_json.get(var_name) != expected_val:
                        return False
            except Exception:
                return False

        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Metrics
# =============================================================================

def compute_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein similarity (1 - normalized distance).

    Returns value in [0, 1] where 1 means identical strings.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    distance = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len)


def compute_codebleu_score(reference: str, candidate: str) -> tuple[float, bool]:
    """
    Compute CodeBLEU score for code comparison.

    Returns value in [0, 1] where 1 means perfect match.
    """
    handler = CodeBleuWarningCapture()
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.WARNING)
    try:
        if not reference.strip() or not candidate.strip():
            return 0.0, False
        result = calc_codebleu(
            references=[[reference]],
            predictions=[candidate],
            lang="c",
            weights=(0.25, 0.25, 0.25, 0.25),
            tokenizer=None
        )
        warning_texts = (
            'EMPTY_NORMALIZED_REF_DFG'
        )
        has_bad_warning = any(
            any(warning_text in message for warning_text in warning_texts)
            for message in handler.messages
        )
        return result['codebleu'], has_bad_warning
    except Exception:
        return 0.0, True
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)

    # return any(WARNING_TEXT in message for message in handler.messages)


def compute_all_metrics(
    generated_code: str,
    original_code: str,
    c_func: str,
    c_test: str,
    real_io_pairs: Optional[dict] = None,
    real_iospec: Optional[str] = None,
    skip_codebleu: bool = False,
) -> dict:
    """
    Compute all evaluation metrics for a generated code sample.

    Args:
        generated_code: The decompiled/generated code
        original_code: The original source code (ground truth)
        c_func: Original function with includes
        c_test: Test code

    Returns:
        Dictionary with metrics: compiles, passes_tests, levenshtein, codebleu
    """
    compiles = check_compilation(generated_code, c_func, c_test)

    passes_tests = False
    if compiles and real_io_pairs is not None and real_iospec is not None:
        passes_tests = run_functional_test(
            generated_code,
            c_func,
            c_test,
            real_io_pairs,
            real_iospec,
            timeout=10,
        )

    original_clean = extract_function_body(original_code)
    generated_clean = extract_function_body(generated_code)

    lev_sim = compute_levenshtein_similarity(original_clean, generated_clean)
    bad_codebleu = False
    codebleu = None
    if not skip_codebleu:
        codebleu, bad_codebleu = compute_codebleu_score(original_clean, generated_clean)
        if bad_codebleu:
            codebleu = None

    return {
        'compiles': compiles,
        'passes_tests': passes_tests,
        'levenshtein': lev_sim,
        'codebleu': codebleu,
        'bad_codebleu': bad_codebleu,
    }
