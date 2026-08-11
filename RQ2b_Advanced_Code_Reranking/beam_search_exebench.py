#!/usr/bin/env python3
"""
Evaluate Transformer beam search on ExeBench.

This keeps the same high-level structure as the original HumanEval-style beam
script, but uses the ExeBench validator harness:
- replace the extern "C" include block with generated code
- compile with g++
- execute via run-sandboxed.sh using input/output JSON files
- compare actual outputs against real_io_pairs
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import pickle
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "LLM4Binary/llm4decompile-9b-v2"
BEAM_SIZES = [1, 2, 4, 8, 16, 32]
MAX_NEW_TOKENS = 1024
NUM_SAMPLES = 10
CACHE_DIR = Path("./cache")
DATA_PATH = "/home/stu354543/rq2b/exebench_executable_5k_ghidra.json"
PROJECT_DIR = Path(__file__).resolve().parent
RUN_SANDBOXED = PROJECT_DIR / "run-sandboxed.sh"
EXEBENCH_INCLUDE_PATTERN = r'extern\s+"C"\s*\{[\s\S]*?#include\s+["<][^">]+[">][\s\S]*?\}'


@dataclass
class BeamResult:
    beam_size: int
    generated_code: str
    compiles: bool
    passes_tests: bool
    generation_time: float


def extract_c_code(text: str) -> str:
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


def simplify_type(type_str: str) -> str:
    type_str = str(type_str).lower()
    if "string" in type_str:
        return "string"
    if "bool" in type_str:
        return "bool"
    if "float" in type_str or "double" in type_str:
        return "float"
    if "int" in type_str:
        return "int"
    return "string"


def replace_candidate(code: str, c_test: str) -> Optional[str]:
    try:
        final_content = re.sub(EXEBENCH_INCLUDE_PATTERN, lambda _: code.strip(), c_test)
        if "#include <cstdint>" not in final_content:
            final_content = "#include <cstdint>\n" + final_content
        return final_content
    except Exception:
        return None


def check_compilation(code: str, c_test: str, timeout: float = 10.0) -> bool:
    final_content = replace_candidate(code, c_test)
    if final_content is None:
        return False

    with tempfile.TemporaryDirectory(dir=str(PROJECT_DIR)) as temp_dir:
        src_path = os.path.join(temp_dir, "test.cpp")
        out_path = os.path.join(temp_dir, "test")

        with open(src_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        try:
            result = subprocess.run(
                f"g++ {src_path} -I . -o {out_path}",
                cwd=str(PROJECT_DIR),
                shell=True,
                capture_output=True,
                timeout=timeout,
            )
            return result.returncode == 0
        except Exception:
            return False


def run_functional_test(
    decompiled_code: str,
    c_test: str,
    real_io_pairs,
    real_iospec,
    timeout: float = 30.0,
) -> bool:
    final_content = replace_candidate(decompiled_code, c_test)
    if final_content is None:
        return False

    io_pairs = real_io_pairs if isinstance(real_io_pairs, dict) else json.loads(real_io_pairs)
    iospec = real_iospec if isinstance(real_iospec, dict) else json.loads(real_iospec)
    typemap = iospec.get("typemap", {})
    inputs_data = io_pairs.get("input", [])
    expected_outputs = io_pairs.get("output", [])

    with tempfile.TemporaryDirectory(dir=str(PROJECT_DIR)) as temp_dir:
        src_path = os.path.join(temp_dir, "test.cpp")
        exe_path = os.path.join(temp_dir, "test")
        input_json = os.path.join(temp_dir, "input_tmp.json")
        output_json = os.path.join(temp_dir, "output_tmp.json")

        with open(src_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        try:
            compile_result = subprocess.run(
                f"g++ -O0 {src_path} -I . -o {exe_path}",
                cwd=str(PROJECT_DIR),
                shell=True,
                capture_output=True,
                timeout=10,
            )
            if compile_result.returncode != 0:
                return False
        except Exception:
            return False

        for inp_dict, exp_out in zip(inputs_data, expected_outputs):
            test_case_input = {}
            for var_name, raw_val in zip(inp_dict.get("var", []), inp_dict.get("value", [])):
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

            with open(input_json, "w", encoding="utf-8") as f:
                json.dump(test_case_input, f, indent=2)

            try:
                run_result = subprocess.run(
                    f"{RUN_SANDBOXED} ./test input_tmp.json output_tmp.json",
                    cwd=temp_dir,
                    shell=True,
                    capture_output=True,
                    timeout=timeout,
                )
                if run_result.returncode != 0:
                    return False
            except Exception:
                return False

            try:
                with open(output_json, "r", encoding="utf-8") as f:
                    actual_output_json = json.load(f)
            except Exception:
                return False

            for var_name, raw_expected in zip(exp_out.get("var", []), exp_out.get("value", [])):
                expected_val = json.loads(raw_expected)
                if actual_output_json.get(var_name) != expected_val:
                    return False

        return True


def generate_with_beam_search(model, tokenizer, prompt: str, beam_size: int, max_new_tokens: int = 1024) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start_time = time.time()

    with torch.no_grad():
        if beam_size == 1:
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=inputs.attention_mask,
            )
        else:
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=beam_size,
                do_sample=False,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=inputs.attention_mask,
            )

    generation_time = time.time() - start_time
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated, generation_time


def main():
    print("=" * 70)
    print("BEAM SEARCH EVALUATION FOR LLM4DECOMPILE ON EXEBENCH")
    print("=" * 70)

    CACHE_DIR.mkdir(exist_ok=True)

    print(f"\nLoading dataset from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if NUM_SAMPLES is not None:
        data = data[:NUM_SAMPLES]

    print(f"Evaluating on {len(data)} samples")
    print(f"Beam sizes to test: {BEAM_SIZES}")

    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    results = {beam_size: [] for beam_size in BEAM_SIZES}
    cache_file = CACHE_DIR / "beam_search_results_exebench.pkl"
    start_idx = 0

    if cache_file.exists():
        print(f"\nLoading cached results from {cache_file}")
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
            results = cached_data.get("results", results)
            start_idx = cached_data.get("last_completed_idx", -1) + 1
        print(f"Resuming from sample {start_idx}")

    print("\n" + "=" * 70)
    print("GENERATING WITH DIFFERENT BEAM SIZES")
    print("=" * 70)

    for idx in tqdm(range(start_idx, len(data)), desc="Samples", initial=start_idx, total=len(data)):
        sample = data[idx]
        prompt = f"# This is the assembly code:\n{sample['input_asm_prompt']}\n# What is the source code?\n"

        for beam_size in BEAM_SIZES:
            try:
                generated, gen_time = generate_with_beam_search(model, tokenizer, prompt, beam_size, MAX_NEW_TOKENS)
                code = extract_c_code(generated)
                compiles = check_compilation(code, sample["c_test"])
                passes = False
                if compiles:
                    passes = run_functional_test(
                        code,
                        sample["c_test"],
                        sample.get("real_io_pairs", {}),
                        sample.get("real_iospec", "{}"),
                    )
                result = BeamResult(
                    beam_size=beam_size,
                    generated_code=code,
                    compiles=compiles,
                    passes_tests=passes,
                    generation_time=gen_time,
                )
            except Exception as e:
                print(f"\nError on sample {idx}, beam_size {beam_size}: {e}")
                result = BeamResult(
                    beam_size=beam_size,
                    generated_code="",
                    compiles=False,
                    passes_tests=False,
                    generation_time=0.0,
                )
            results[beam_size].append(result)

        if (idx + 1) % 10 == 0:
            with open(cache_file, "wb") as f:
                pickle.dump({"results": results, "last_completed_idx": idx}, f)

    with open(cache_file, "wb") as f:
        pickle.dump({"results": results, "last_completed_idx": len(data) - 1}, f)

    print("\n" + "=" * 70)
    print("RESULTS: BEAM SEARCH PERFORMANCE")
    print("=" * 70)

    summary = []
    for beam_size in BEAM_SIZES:
        beam_results = results[beam_size]
        n_total = len(beam_results)
        n_compile = sum(1 for r in beam_results if r.compiles)
        n_pass = sum(1 for r in beam_results if r.passes_tests)
        avg_time = sum(r.generation_time for r in beam_results) / max(n_total, 1)
        compile_rate = n_compile / n_total * 100 if n_total > 0 else 0.0
        pass_rate = n_pass / n_total * 100 if n_total > 0 else 0.0

        summary.append({
            "beam_size": beam_size,
            "total": n_total,
            "compile": n_compile,
            "compile_rate": compile_rate,
            "pass": n_pass,
            "pass_rate": pass_rate,
            "avg_time": avg_time,
        })

        beam_label = "greedy" if beam_size == 1 else f"beam={beam_size}"
        print(f"\n  {beam_label:12s}:")
        print(f"    Compilation: {n_compile:4d}/{n_total:4d} = {compile_rate:5.1f}%")
        print(f"    Pass tests:  {n_pass:4d}/{n_total:4d} = {pass_rate:5.1f}%")
        print(f"    Avg time:    {avg_time:.2f}s per sample")

    summary_file = CACHE_DIR / "beam_search_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
