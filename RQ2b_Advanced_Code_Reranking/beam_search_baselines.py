#!/usr/bin/env python3
"""
Greedy decoding and beam-search baselines for the RQ2 pipeline.

This script generates one decoded program per sample for greedy decoding
(beam=1) and configurable beam sizes, evaluates compilation and test pass,
and saves a summary compatible with evaluate_selection.py/export_to_tex.py.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import subprocess
import tempfile
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from tqdm import tqdm

import Levenshtein
from codebleu import calc_codebleu

try:
    from vllm import LLM, SamplingParams
    try:
        from vllm import BeamSearchParams
    except ImportError:
        BeamSearchParams = None
except ImportError:
    LLM = None
    SamplingParams = None
    BeamSearchParams = None


@dataclass
class Config:
    model_name: str = "LLM4Binary/llm4decompile-9b-v2"
    num_samples: int = 10
    beam_sizes: List[int] = None
    max_new_tokens: int = 4096
    max_model_len: int = 12288
    cache_dir: str = "./cache"
    data_path: str = "/home/stu354543/rq2b/exebench_executable_5k_ghidra.json"
    test_step_timeout: int = 20

    def __post_init__(self):
        if self.beam_sizes is None:
            self.beam_sizes = [1, 2, 4, 8, 16, 32]


@dataclass
class Sample:
    idx: int
    ghidra: str
    c_func: str
    c_test: str
    real_io_pairs: dict
    real_iospec: str


PROJECT_DIR = Path(__file__).resolve().parent
EXEBENCH_INCLUDE_PATTERN = r'extern\s+"C"\s*\{[\s\S]*?#include\s+["<][^">]+[">][\s\S]*?\}'


def load_dataset(data_path: str) -> List[Dict]:
    with open(data_path, "r") as f:
        return json.load(f)


def coerce_real_io_pairs(real_io_pairs) -> dict:
    if isinstance(real_io_pairs, dict):
        return real_io_pairs
    if isinstance(real_io_pairs, str) and real_io_pairs.strip():
        try:
            return json.loads(real_io_pairs)
        except Exception:
            return {}
    return {}


def parse_real_iospec(real_iospec) -> dict:
    if isinstance(real_iospec, dict):
        return real_iospec
    if isinstance(real_iospec, str) and real_iospec.strip():
        try:
            return json.loads(real_iospec)
        except Exception:
            return {}
    return {}


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


def replace_exebench_include_with_candidate(code: str, c_test: str) -> Optional[str]:
    try:
        final_content = re.sub(EXEBENCH_INCLUDE_PATTERN, lambda _: code.strip(), c_test)
        if "#include <cstdint>" not in final_content:
            final_content = "#include <cstdint>\n" + final_content
        return final_content
    except Exception:
        return None


def evaluate_candidate(code: str, sample: Sample) -> Tuple[bool, bool]:
    if not code.strip():
        return False, False

    final_content = replace_exebench_include_with_candidate(code, sample.c_test)
    if final_content is None:
        return False, False

    io_pairs = coerce_real_io_pairs(sample.real_io_pairs)
    iospec = parse_real_iospec(sample.real_iospec)
    typemap = iospec.get("typemap", {})
    inputs_data = io_pairs.get("input", [])
    expected_outputs = io_pairs.get("output", [])

    with tempfile.TemporaryDirectory(dir=str(PROJECT_DIR)) as temp_dir:
        c_file = os.path.join(temp_dir, "test.cpp")
        exe = os.path.join(temp_dir, "test")
        input_json = os.path.join(temp_dir, "input_tmp.json")
        output_json = os.path.join(temp_dir, "output_tmp.json")
        with open(c_file, "w", encoding="utf-8") as f:
            f.write(final_content)

        try:
            compile_result = subprocess.run(
                f"g++ -O0 {c_file} -I . -o {exe}",
                cwd=str(PROJECT_DIR),
                shell=True,
                capture_output=True,
                timeout=10,
            )
        except Exception:
            return False, False
        if compile_result.returncode != 0:
            return False, False

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
                    f'{os.path.join(str(PROJECT_DIR), "run-sandboxed.sh")} ./test input_tmp.json output_tmp.json',
                    cwd=temp_dir,
                    shell=True,
                    capture_output=True,
                    timeout=Config().test_step_timeout,
                )
            except Exception:
                return True, False

            if run_result.returncode != 0:
                return True, False

            try:
                with open(output_json, "r", encoding="utf-8") as f:
                    actual_output_json = json.load(f)
            except Exception:
                return True, False

            for var_name, raw_expected in zip(exp_out.get("var", []), exp_out.get("value", [])):
                if actual_output_json.get(var_name) != json.loads(raw_expected):
                    return True, False

        return True, True


def extract_function_body(code: str) -> str:
    lines = []
    for line in code.split("\n"):
        if line.strip().startswith("#include"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def compute_levenshtein_similarity(reference: str, candidate: str) -> float:
    if not reference and not candidate:
        return 1.0
    if not reference or not candidate:
        return 0.0
    distance = Levenshtein.distance(reference, candidate)
    max_len = max(len(reference), len(candidate))
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


def generate_for_beam_size(llm: LLM, samples: List[Sample], config: Config, beam_size: int) -> Tuple[Dict[int, str], float]:
    tokenizer = llm.get_tokenizer()
    max_prompt_len = config.max_model_len - config.max_new_tokens

    valid_samples = []
    prompts = []
    outputs_by_idx: Dict[int, str] = {}

    for sample in samples:
        prompt = f"# This is the assembly code:\n{sample.ghidra}\n# What is the source code?\n"
        token_len = len(tokenizer.encode(prompt))
        if token_len <= max_prompt_len:
            valid_samples.append(sample)
            prompts.append(prompt)

    print(f"  Valid samples: {len(valid_samples)}/{len(samples)}")
    start_time = time.time()

    if beam_size == 1:
        params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=config.max_new_tokens,
        )
        request_outputs = llm.generate(prompts, params)
        for sample, output in zip(valid_samples, request_outputs):
            if output.outputs:
                outputs_by_idx[sample.idx] = output.outputs[0].text
        return outputs_by_idx, time.time() - start_time

    if BeamSearchParams is not None and hasattr(llm, "beam_search"):
        try:
            beam_params = BeamSearchParams(
                beam_width=beam_size,
                max_tokens=config.max_new_tokens,
                temperature=0.0,
            )
            request_outputs = llm.beam_search(prompts, beam_params)
            for sample, output in zip(valid_samples, request_outputs):
                if hasattr(output, "outputs") and output.outputs:
                    outputs_by_idx[sample.idx] = output.outputs[0].text
                elif hasattr(output, "sequences") and output.sequences:
                    seq0 = output.sequences[0]
                    outputs_by_idx[sample.idx] = getattr(seq0, "text", str(seq0))
                elif isinstance(output, list) and output:
                    seq0 = output[0]
                    outputs_by_idx[sample.idx] = getattr(seq0, "text", str(seq0))
            return outputs_by_idx, time.time() - start_time
        except Exception as exc:
            print(f"  Native beam_search API failed for beam={beam_size}: {exc}")

    # Compatibility fallback: request multiple sampled completions and keep the
    # first-ranked one. This is not true beam search, but it keeps the baseline
    # runnable on vLLM builds without exposed beam-search support.
    print(f"  Falling back to sampled multi-output decoding for beam={beam_size}")
    params = SamplingParams(
        n=beam_size,
        temperature=0.8,
        top_p=0.95,
        max_tokens=config.max_new_tokens,
        seed=0,
    )
    request_outputs = llm.generate(prompts, params)
    for sample, output in zip(valid_samples, request_outputs):
        if output.outputs:
            outputs_by_idx[sample.idx] = output.outputs[0].text

    return outputs_by_idx, time.time() - start_time


def evaluate_outputs(samples: List[Sample], outputs_by_idx: Dict[int, str], beam_size: int, generation_time: float) -> Dict:
    total = len(samples)
    compiled = 0
    passed = 0
    lev_sum = 0.0
    codebleu_sum = 0.0

    for sample in tqdm(samples, desc=f"Evaluating beam={beam_size}"):
        code = outputs_by_idx.get(sample.idx, "")
        compile_ok, pass_ok = evaluate_candidate(code, sample)
        if compile_ok:
            compiled += 1
        if pass_ok:
            passed += 1

        reference = extract_function_body(sample.c_func)
        candidate = extract_function_body(code)
        lev_sum += compute_levenshtein_similarity(reference, candidate)
        codebleu_sum += compute_codebleu(reference, candidate)

    return {
        "beam_size": beam_size,
        "total": total,
        "compile": compiled,
        "correct": passed,
        "pass": passed,
        "pass_rate": 100.0 * passed / total if total else 0.0,
        "compile_rate": 100.0 * compiled / total if total else 0.0,
        "avg_time": generation_time / total if total else 0.0,
        "levenshtein": 100.0 * lev_sum / total if total else 0.0,
        "codebleu": 100.0 * codebleu_sum / total if total else 0.0,
    }


def main():
    if LLM is None or SamplingParams is None:
        raise RuntimeError("vLLM is not available in this environment.")

    config = Config()
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("GREEDY / BEAM SEARCH BASELINES")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Samples: {config.num_samples}")
    print(f"  Beam sizes: {config.beam_sizes}")
    print(f"  Max new tokens: {config.max_new_tokens}")
    print(f"  Max model length: {config.max_model_len}")

    dataset = load_dataset(config.data_path)

    samples = []
    for i in range(config.num_samples):
        row = dataset[i]
        samples.append(Sample(
            idx=i,
            ghidra=row["input_asm_prompt"],
            c_func=row["c_func"],
            c_test=row["c_test"],
            real_io_pairs=row.get("real_io_pairs", {}),
            real_iospec=row.get("real_iospec", ""),
        ))

    print(f"\nLoaded {len(samples)} samples")
    print("\nLoading LLM...")
    llm = LLM(
        model=config.model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=config.max_model_len,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
    )

    summary = []
    for beam_size in config.beam_sizes:
        print(f"\n=== Decoding with beam={beam_size} ===")
        outputs_cache = cache_dir / f"beam_outputs_{beam_size}.pkl"
        if outputs_cache.exists():
            print(f"Loading cached beam outputs from {outputs_cache}")
            with open(outputs_cache, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and "outputs_by_idx" in cached:
                outputs_by_idx = cached["outputs_by_idx"]
                generation_time = float(cached.get("generation_time", 0.0))
            else:
                outputs_by_idx = cached
                generation_time = 0.0
        else:
            outputs_by_idx, generation_time = generate_for_beam_size(llm, samples, config, beam_size)
            with open(outputs_cache, "wb") as f:
                pickle.dump({
                    "outputs_by_idx": outputs_by_idx,
                    "generation_time": generation_time,
                }, f)
            print(f"Saved beam outputs to {outputs_cache}")

        beam_result = evaluate_outputs(samples, outputs_by_idx, beam_size, generation_time)
        summary.append(beam_result)
        print(
            f"  beam={beam_size}: pass@1={beam_result['pass_rate']:.1f}% "
            f"compile={beam_result['compile_rate']:.1f}%"
        )

    output_file = cache_dir / "beam_search_summary.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved beam summary to {output_file}")


if __name__ == "__main__":
    main()
