#!/usr/bin/env python3
"""
Export RQ2 evaluation results to LaTeX table format.
"""

import json
from pathlib import Path


def load_results():
    """Load evaluation results from JSON."""
    with open("selection_evaluation_results.json", "r") as f:
        return json.load(f)


def load_beam_results():
    """Load beam search results."""
    beam_file = Path("cache/beam_search_summary.json")
    if beam_file.exists():
        with open(beam_file, "r") as f:
            return json.load(f)
    return None


def format_method_name(method: str) -> str:
    """Format method name for display in LaTeX."""
    name_map = {
        "oracle": "Oracle (upper bound)",
        "random": "Random",
        "mbr_consensus": "MBR Consensus",
        "execution_guided": "Execution-Guided",
        "ensemble_unixcoder_ft": "Ensemble (UniXcoder-FT)",
        "ensemble_text": "Ensemble (Text)",
        "edit_similarity": "Edit Similarity",
        "token_f1": "Token F1",
        "bigram_sim": "Bigram Similarity",
        "trigram_sim": "Trigram Similarity",
        "ast_node_sim": "AST Node Similarity",
        "control_flow_sim": "Control Flow Similarity",
        "identifier_sim": "Identifier Similarity",
        "unixcoder_base": "UniXcoder (Base)",
        "unixcoder_finetuned": "UniXcoder (Fine-tuned)",
    }
    return name_map.get(method, method)


def generate_main_table(results: dict) -> str:
    """Generate main results table."""
    best_non_oracle = max(
        (item for item in results["results"] if item["method"] != "oracle"),
        key=lambda x: x["pass_rate"],
        default=None,
    )
    best_pass_method = best_non_oracle["method"] if best_non_oracle else None
    best_codebleu_item = max(
        (item for item in results["results"] if item["method"] != "oracle"),
        key=lambda x: x["codebleu"],
        default=None,
    )
    best_codebleu_method = best_codebleu_item["method"] if best_codebleu_item else None

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{RQ2: Selection Algorithm Evaluation Results}",
        r"\label{tab:rq2-selection}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Pass@1} & \textbf{Levenshtein} & \textbf{CodeBLEU} \\",
        r"\midrule",
    ]

    for item in results["results"]:
        method = format_method_name(item["method"])
        pass_rate = item["pass_rate"]
        lev = item["levenshtein"]
        codebleu = item["codebleu"]

        # Highlight dynamic winners instead of hardcoding a specific method
        if item["method"] == best_pass_method or item["method"] == best_codebleu_method:
            method_fmt = f"\\textbf{{{method}}}"
            pass_fmt = f"\\textbf{{{pass_rate:.1f}\\%}}" if item["method"] == best_pass_method else f"{pass_rate:.1f}\\%"
            lev_fmt = f"{lev:.1f}\\%"
            codebleu_fmt = f"\\textbf{{{codebleu:.1f}\\%}}" if item["method"] == best_codebleu_method else f"{codebleu:.1f}\\%"
            line = f"{method_fmt} & {pass_fmt} & {lev_fmt} & {codebleu_fmt} \\\\"
        elif item["method"] == "oracle":
            line = f"\\textit{{{method}}} & \\textit{{{pass_rate:.1f}\\%}} & \\textit{{{lev:.1f}\\%}} & \\textit{{{codebleu:.1f}\\%}} \\\\"
        else:
            line = f"{method} & {pass_rate:.1f}\\% & {lev:.1f}\\% & {codebleu:.1f}\\% \\\\"

        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_compact_table(results: dict) -> str:
    """Generate compact table with top methods only."""
    best_non_oracle = max(
        (item for item in results["results"] if item["method"] != "oracle"),
        key=lambda x: x["pass_rate"],
        default=None,
    )
    best_pass_method = best_non_oracle["method"] if best_non_oracle else None
    best_codebleu_item = max(
        (item for item in results["results"] if item["method"] != "oracle"),
        key=lambda x: x["codebleu"],
        default=None,
    )
    best_codebleu_method = best_codebleu_item["method"] if best_codebleu_item else None

    top_methods = ["oracle", "mbr_consensus", "execution_guided", "random", "trigram_sim",
                   "unixcoder_finetuned", "ensemble_unixcoder_ft"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{RQ2: Top Selection Methods Comparison}",
        r"\label{tab:rq2-top-methods}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Pass@1} & \textbf{Levenshtein} & \textbf{CodeBLEU} \\",
        r"\midrule",
    ]

    for item in results["results"]:
        if item["method"] not in top_methods:
            continue

        method = format_method_name(item["method"])
        pass_rate = item["pass_rate"]
        lev = item["levenshtein"]
        codebleu = item["codebleu"]

        if item["method"] == best_pass_method or item["method"] == best_codebleu_method:
            method_fmt = f"\\textbf{{{method}}}"
            pass_fmt = f"\\textbf{{{pass_rate:.1f}\\%}}" if item["method"] == best_pass_method else f"{pass_rate:.1f}\\%"
            lev_fmt = f"{lev:.1f}\\%"
            codebleu_fmt = f"\\textbf{{{codebleu:.1f}\\%}}" if item["method"] == best_codebleu_method else f"{codebleu:.1f}\\%"
            line = f"{method_fmt} & {pass_fmt} & {lev_fmt} & {codebleu_fmt} \\\\"
        elif item["method"] == "oracle":
            line = f"\\textit{{{method}}} & \\textit{{{pass_rate:.1f}\\%}} & \\textit{{{lev:.1f}\\%}} & \\textit{{{codebleu:.1f}\\%}} \\\\"
        else:
            line = f"{method} & {pass_rate:.1f}\\% & {lev:.1f}\\% & {codebleu:.1f}\\% \\\\"

        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_beam_table(beam_results: list) -> str:
    """Generate beam search comparison table."""
    # Check if new metrics are available
    has_new_metrics = 'levenshtein' in beam_results[0] if beam_results else False

    if has_new_metrics:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Beam Search Performance Degradation}",
            r"\label{tab:rq2-beam-search}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"\textbf{Method} & \textbf{Pass@1} & \textbf{Levenshtein} & \textbf{CodeBLEU} & \textbf{Compile} \\",
            r"\midrule",
        ]

        for beam in beam_results:
            beam_size = beam["beam_size"]
            pass_rate = beam["pass_rate"]
            compile_rate = beam["compile_rate"]
            lev = beam.get("levenshtein", 0)
            codebleu = beam.get("codebleu", 0)

            if beam_size == 1:
                label = "Greedy (beam=1)"
            else:
                label = f"Beam search (k={beam_size})"

            line = f"{label} & {pass_rate:.1f}\\% & {lev:.1f}\\% & {codebleu:.1f}\\% & {compile_rate:.1f}\\% \\\\"
            lines.append(line)
    else:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Beam Search Performance Degradation}",
            r"\label{tab:rq2-beam-search}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"\textbf{Method} & \textbf{Pass@1} & \textbf{Compile Rate} \\",
            r"\midrule",
        ]

        for beam in beam_results:
            beam_size = beam["beam_size"]
            pass_rate = beam["pass_rate"]
            compile_rate = beam["compile_rate"]

            if beam_size == 1:
                label = "Greedy (beam=1)"
            else:
                label = f"Beam search (k={beam_size})"

            line = f"{label} & {pass_rate:.1f}\\% & {compile_rate:.1f}\\% \\\\"
            lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_combined_table(results: dict, beam_results: list) -> str:
    """Generate combined table comparing selection vs beam search."""
    best_non_oracle = max(
        (item for item in results["results"] if item["method"] not in ["oracle", "random"]),
        key=lambda x: x["pass_rate"],
        default=None,
    )
    best_pass_method = best_non_oracle["method"] if best_non_oracle else None

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Selection vs Beam Search Comparison}",
        r"\label{tab:rq2-combined}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Pass@1} \\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{Selection Methods (200 samples)}} \\",
    ]

    # Add top selection methods
    top_selection = ["mbr_consensus", "ensemble_unixcoder_ft", "trigram_sim", "random"]
    for item in results["results"]:
        if item["method"] not in top_selection:
            continue
        method = format_method_name(item["method"])
        pass_rate = item["pass_rate"]
        if item["method"] == best_pass_method:
            line = f"\\quad \\textbf{{{method}}} & \\textbf{{{pass_rate:.1f}\\%}} \\\\"
        else:
            line = f"\\quad {method} & {pass_rate:.1f}\\% \\\\"
        lines.append(line)

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\textit{Beam Search Methods}} \\")

    # Add beam search results
    for beam in beam_results:
        beam_size = beam["beam_size"]
        pass_rate = beam["pass_rate"]
        if beam_size == 1:
            label = "Greedy (beam=1)"
        else:
            label = f"Beam search (k={beam_size})"
        line = f"\\quad {label} & {pass_rate:.1f}\\% \\\\"
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    results = load_results()
    beam_results = load_beam_results()

    print("=" * 70)
    print("LATEX TABLES FOR RQ2 EVALUATION")
    print("=" * 70)

    print("\n% === FULL RESULTS TABLE ===\n")
    print(generate_main_table(results))

    print("\n\n% === COMPACT TABLE (TOP METHODS) ===\n")
    print(generate_compact_table(results))

    if beam_results:
        print("\n\n% === BEAM SEARCH TABLE ===\n")
        print(generate_beam_table(beam_results))

        print("\n\n% === COMBINED COMPARISON TABLE ===\n")
        print(generate_combined_table(results, beam_results))

    print("\n" + "=" * 70)
    print("Copy the tables above into your LaTeX document.")
    print("Required packages: \\usepackage{booktabs}")
    print("=" * 70)


if __name__ == "__main__":
    main()
