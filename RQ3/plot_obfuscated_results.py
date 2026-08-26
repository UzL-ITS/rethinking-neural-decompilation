#!/usr/bin/env python3
"""
Plot obfuscated oracle results as a heatmap.

Reads results from cache_oracle/oracle_results.json and creates:
1. A heatmap with obfuscation types as rows and optimization levels as columns
2. Bar charts for individual breakdowns

Run this after analyze_obfuscated_oracle.py has completed.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).resolve().parent


def load_results(cache_dir: str = None) -> dict:
    """Load results from JSON file."""
    if cache_dir is None:
        cache_dir = str(_SCRIPT_DIR / "cache_oracle")

    results_file = Path(cache_dir) / "oracle_results.json"

    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file}\n"
            "Run analyze_obfuscated_oracle.py first."
        )

    with open(results_file, 'r') as f:
        return json.load(f)


def plot_heatmap(results: dict, output_path: str = None):
    """Create a heatmap of pass@256 rates by obfuscation and optimization level."""

    if output_path is None:
        output_path = str(_SCRIPT_DIR / "plots/obfuscated_heatmap.png")

    heatmap = results['heatmap']

    obf_types = sorted(heatmap.keys())
    opt_levels = ['O0', 'O1', 'O2', 'O3']

    matrix = np.zeros((len(obf_types), len(opt_levels)))
    annotations = [['' for _ in opt_levels] for _ in obf_types]

    for i, obf in enumerate(obf_types):
        for j, opt in enumerate(opt_levels):
            if opt in heatmap[obf]:
                pct = heatmap[obf][opt]['percentage']
                passed = heatmap[obf][opt]['passed']
                total = heatmap[obf][opt]['total']
                matrix[i, j] = pct
                annotations[i][j] = f'{pct:.1f}%\n({passed}/{total})'

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Pass@256 Rate (%)', rotation=-90, va="bottom", fontsize=12)

    ax.set_xticks(np.arange(len(opt_levels)))
    ax.set_yticks(np.arange(len(obf_types)))
    ax.set_xticklabels(opt_levels, fontsize=12)
    ax.set_yticklabels(obf_types, fontsize=12)

    for i in range(len(obf_types)):
        for j in range(len(opt_levels)):
            text_color = 'white' if matrix[i, j] < 50 else 'black'
            ax.text(j, i, annotations[i][j], ha='center', va='center',
                    color=text_color, fontsize=10, fontweight='bold')

    ax.set_xlabel('Optimization Level', fontsize=14)
    ax.set_ylabel('Obfuscation Type', fontsize=14)
    ax.set_title('LLM4Decompile Pass@256 on Obfuscated Code', fontsize=16, fontweight='bold')

    ax.set_xticks(np.arange(len(opt_levels)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(obf_types)+1)-.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()


def plot_bar_charts(results: dict, output_path: str = None):
    """Create bar charts for obfuscation and optimization breakdowns."""

    if output_path is None:
        output_path = str(_SCRIPT_DIR / "plots/obfuscated_bars.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    obf_data = results['by_obfuscation']
    obf_types = sorted(obf_data.keys())
    obf_pcts = [obf_data[o]['percentage'] for o in obf_types]

    bars1 = ax1.bar(obf_types, obf_pcts, color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'])
    ax1.set_xlabel('Obfuscation Type', fontsize=12)
    ax1.set_ylabel('Pass@256 Rate (%)', fontsize=12)
    ax1.set_title('By Obfuscation Type', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)

    for bar, pct in zip(bars1, obf_pcts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2 = axes[1]
    opt_data = results['by_optimization']
    opt_levels = ['O0', 'O1', 'O2', 'O3']
    opt_pcts = [opt_data[o]['percentage'] for o in opt_levels]

    bars2 = ax2.bar(opt_levels, opt_pcts, color=['#1abc9c', '#f39c12', '#e67e22', '#c0392b'])
    ax2.set_xlabel('Optimization Level', fontsize=12)
    ax2.set_ylabel('Pass@256 Rate (%)', fontsize=12)
    ax2.set_title('By Optimization Level', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)

    for bar, pct in zip(bars2, opt_pcts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    overall = results['overall']
    fig.suptitle(
        f"LLM4Decompile Pass@256 Analysis\nOverall: {overall['passed']}/{overall['total']} = {overall['percentage']:.1f}%",
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved bar charts to {output_path}")
    plt.close()


def plot_detailed_analysis(results: dict, output_path: str = None):
    """Create a detailed analysis plot showing compilation vs pass rates."""

    if output_path is None:
        output_path = str(_SCRIPT_DIR / "plots/obfuscated_detailed.png")

    detailed = results['detailed_results']

    stats = defaultdict(lambda: {'compile_rate': [], 'pass_rate': []})

    for r in detailed:
        key = (r['obf_pass'], r['opt_level'])
        if r['num_candidates'] > 0:
            compile_rate = 100 * r['num_compiling'] / r['num_candidates']
            pass_rate = 100 * r['num_passing'] / r['num_candidates'] if r['num_compiling'] > 0 else 0
            stats[key]['compile_rate'].append(compile_rate)
            stats[key]['pass_rate'].append(pass_rate)

    obf_types = sorted(set(r['obf_pass'] for r in detailed))
    opt_levels = ['O0', 'O1', 'O2', 'O3']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, obf in enumerate(obf_types):
        ax = axes[idx // 2, idx % 2]

        compile_rates = []
        pass_rates = []
        oracle_rates = []

        for opt in opt_levels:
            key = (obf, opt)
            if key in stats:
                compile_rates.append(np.mean(stats[key]['compile_rate']))
                pass_rates.append(np.mean(stats[key]['pass_rate']))
                subset = [r for r in detailed if r['obf_pass'] == obf and r['opt_level'] == opt]
                oracle = sum(1 for r in subset if r['oracle_pass']) / len(subset) * 100
                oracle_rates.append(oracle)
            else:
                compile_rates.append(0)
                pass_rates.append(0)
                oracle_rates.append(0)

        x = np.arange(len(opt_levels))
        width = 0.25

        ax.bar(x - width, compile_rates, width, label='Avg Compile Rate', color='#3498db')
        ax.bar(x, pass_rates, width, label='Avg Pass Rate', color='#2ecc71')
        ax.bar(x + width, oracle_rates, width, label='Oracle (Pass@256)', color='#e74c3c')

        ax.set_xlabel('Optimization Level')
        ax.set_ylabel('Rate (%)')
        ax.set_title(f'Obfuscation: {obf}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(opt_levels)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Detailed Analysis by Obfuscation Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved detailed analysis to {output_path}")
    plt.close()


def print_failed_samples(results: dict):
    """Print samples that failed all 256 attempts."""

    detailed = results['detailed_results']
    failed = [r for r in detailed if not r['oracle_pass']]

    print(f"\n{'='*70}")
    print(f"FAILED SAMPLES ({len(failed)} total)")
    print(f"{'='*70}")

    by_task = defaultdict(list)
    for r in failed:
        by_task[r['task_id']].append((r['obf_pass'], r['opt_level']))

    print(f"\nUnique functions that failed in at least one config: {len(by_task)}")

    all_tasks = set(r['task_id'] for r in detailed)
    always_fail = []
    for task_id in all_tasks:
        task_results = [r for r in detailed if r['task_id'] == task_id]
        if all(not r['oracle_pass'] for r in task_results):
            always_fail.append(task_id)

    print(f"Functions that ALWAYS fail (all obf/opt combos): {len(always_fail)}")
    if always_fail:
        print(f"  Task IDs: {always_fail[:20]}{'...' if len(always_fail) > 20 else ''}")


def main():
    print("Loading results...")
    results = load_results()

    print(f"\nOverall pass@256: {results['overall']['percentage']:.1f}%")

    print("\nGenerating plots...")
    plot_heatmap(results)
    plot_bar_charts(results)
    plot_detailed_analysis(results)

    print_failed_samples(results)

    print("\nDone! Generated:")
    print("  - plots/obfuscated_heatmap.png")
    print("  - plots/obfuscated_bars.png")
    print("  - plots/obfuscated_detailed.png")


if __name__ == "__main__":
    main()
