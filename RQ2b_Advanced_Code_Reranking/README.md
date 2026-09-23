# RQ2b: Advanced Candidate Reranking for LLM4Decompile

**Research question:** How effective are advanced selection and reranking
strategies, beyond model perplexity, at improving pass@1 when multiple
decompilation candidates are available?

## Objective Status

- [x] Review at least two code or sequence reranking approaches.
- [x] Implement MBR consensus reranking.
- [x] Implement execution-guided reranking.
- [x] Implement representation-based correctness scoring (RepE).
- [x] Compare against perplexity, majority voting, text, AST, embedding,
  random, and oracle baselines.
- [x] Evaluate the selectors on C1, C2, C3, and C4.

The checklist records what was completed in the experiment. It does not imply
that every server-side driver, candidate cache, and intermediate result is
currently included in this repository. See [Repository Coverage](#repository-coverage).

## Experimental Configurations

| ID | Decompiler input | Dataset | Generation model | Samples |
|---|---|---|---|---:|
| C1 | Ghidra pseudo-code | HumanEval-Decompile | `LLM4Binary/llm4decompile-9b-v2` | 656 |
| C2 | Assembly | HumanEval-Decompile | `LLM4Binary/llm4decompile-6.7b-v1.5` | 656 |
| C3 | Ghidra pseudo-code | ExeBench | `LLM4Binary/llm4decompile-9b-v2` | 1,000 |
| C4 | Assembly | ExeBench | `LLM4Binary/llm4decompile-6.7b-v1.5` | 1,000 |

The experiment uses 100 cached candidates per sample. Candidates were produced
with temperature sampling using temperatures `0.0`, `0.3`, `0.5`, `0.7`,
`0.9`, and `1.0`, with a maximum sequence length of 4,096 tokens.

The candidate pool `P` for reranking contains only candidates that compile,
unless a selector explicitly states otherwise. Functional correctness is
measured by compiling and running the selected candidate against the complete
test harness. Levenshtein similarity and CodeBLEU are also reported.

## Research Basis

The following research directions were reviewed for this task:

| Direction | Relevant work | Use in this experiment |
|---|---|---|
| Minimum Bayes Risk | *Minimum Bayes-Risk Decoding for Statistical Machine Translation*; *Is MAP Decoding All You Need? The Inadequacy of the Mode in Neural Machine Translation* | Implemented as candidate-consensus reranking |
| Execution-guided selection | *Natural Language to Code Translation with Execution* | Implemented using partial test checkpoints |
| Internal correctness representations | *On LLMs' Internal Representation of Code Correctness* | Implemented as `repe_correctness` |
| Learned code ranking | *Fault-Aware Neural Code Rankers* | Reviewed, but not included in the reported C1-C4 table because it requires a separately trained fault-aware ranker |

The implemented advanced selectors are therefore MBR consensus,
execution-guided reranking, and RepE correctness. Perplexity remains a static
baseline rather than the proposed advanced method.

## Execution Pipeline

All configurations follow the same high-level procedure:

```text
1. Generate or load 100 candidates for every sample
2. Compile every candidate
3. Form the compiling candidate pool P
4. Compute static selector scores
5. Compute execution-guided scores where a harness is available
6. Run complete functional tests and cache the ground-truth results
7. Select one candidate per method
8. Report pass@1, Levenshtein similarity, and CodeBLEU
```

The phase caches use the following names:

```text
candidates_<samples>_100.pkl       Candidate programs and sample metadata
phase1_compilation.pkl             Compilation status
phase2_text_metrics.pkl            Text and AST scores
phase2_5_mbr.pkl                   MBR consensus scores
phase3_unixcoder.pkl               Base and fine-tuned UniXcoder scores
phase3_5_execution_guided.pkl      Partial-execution scores
phase3_perplexity.pkl              Negative-perplexity scores
phase3_7_repe_correctness.pkl      RepE correctness scores
phase4_test_results.pkl            Complete functional-test results
selection_evaluation_results.json  Final selector evaluation
```

### HumanEval-Decompile: C1 and C2

HumanEval candidates are compiled as C programs with `gcc`. The generated
function is combined with the sample's assertion-based test harness.
Execution-guided reranking constructs up to three partial harnesses by retaining
prefixes of the available assertions. A candidate's score is the fraction of
these checkpoints it passes.

The relevant implementation files are:

```text
selection_pipeline_mbr.py          Candidate loading/generation, compilation, MBR, tests
compute_execution_guided_cache.py  HumanEval partial-checkpoint execution scores
evaluate_selection.py              Final selection and evaluation from phase caches
```

Candidate PKLs and phase caches are generated during execution and therefore
do not need to be committed to the repository. `evaluate_selection.py` expects
them under its configured `CACHE_DIR`. C1 and C2 must use separate cache
directories; phase caches from the two configurations must not be mixed.

### ExeBench: C3 and C4

ExeBench uses generated C/C++ functions, `real_io_pairs`, and `real_iospec`.
For every candidate, the pipeline:

```text
1. Substitutes the candidate into the ExeBench C++ validator harness
2. Compiles the validator with g++
3. Executes the compiled validator through run-sandboxed.sh
4. Reads the generated output JSON
5. Compares it with the expected output in real_io_pairs
```

The reported C3/C4 experiment used this ExeBench harness logic for advanced
reranking. However, the comprehensive ExeBench reranking driver used to create
all phase caches is not included in the current repository listing.
`beam_search_exebench.py` contains the ExeBench harness and sandboxed execution
logic for the beam-search baseline, but it is not a replacement for the missing
100-candidate MBR/execution-guided evaluator.

The sandbox backend is controlled with `SANDBOX_BACKEND`. The supported values
in `run-sandboxed.sh` are `systemd`, `bwrap`, `auto`, and `none`. ExeBench
evaluation should use `bwrap` or another available isolation backend, not
`none`, because model-generated code is executed.

## Selection Strategies

| Category | Selector | Description |
|---|---|---|
| Upper bound | `oracle` | Returns a passing candidate when one exists; not deployable |
| Baseline | `random` | Uniformly samples one candidate from `P` |
| Likelihood | `neg_perplexity` | Ranks by negative candidate perplexity conditioned on the decompiler input |
| Consensus | `mbr_consensus` | Selects the candidate with the highest expected lexical and structural agreement with `P` |
| Voting | majority vote | Selects the most frequent exact or normalized candidate, over compiling candidates or all candidates |
| Execution | `execution_guided` | Ranks by the fraction of partial test checkpoints passed |
| Representation | `repe_correctness` | Projects model hidden states onto a fitted internal correctness direction |
| Text | `edit_similarity` | Normalized character-level Levenshtein similarity to the decompiler input |
| Text | `token_f1` | Token-set overlap F1 |
| Text | `bigram_sim`, `trigram_sim` | Token n-gram Jaccard similarity |
| AST | `ast_node_sim` | Cosine similarity between AST node-type count vectors |
| AST | `control_flow_sim` | Agreement between selected control-flow node counts |
| AST | `identifier_sim` | Jaccard similarity between identifier sets |
| Embedding | `unixcoder_base` | Cosine similarity from pretrained UniXcoder embeddings |
| Embedding | `unixcoder_finetuned` | Cosine similarity from contrastively fine-tuned UniXcoder embeddings |
| Ensemble | `ensemble_text` | `0.4*bigram + 0.4*trigram + 0.2*token_f1` |
| Ensemble | `ensemble_unixcoder_ft` | `0.5*unixcoder_ft + 0.25*bigram + 0.25*trigram` |

MBR uses a code-aware pairwise utility:

```text
0.40 * token multiset F1
+ 0.30 * bigram Jaccard similarity
+ 0.20 * AST node-count cosine similarity
+ 0.10 * control-flow agreement
```

RepE differs from execution-guided selection. Test outcomes are needed offline
to fit and validate the correctness direction, but scoring a new candidate uses
hidden-state projections and does not execute that candidate.

## Results

Values are pass@1 percentages reported as mean +/- standard deviation.

| Selector | C1 | C2 | C3 | C4 | Avg. |
|---|---:|---:|---:|---:|---:|
| Oracle | 92.7 +/- 1.0 | 80.4 +/- 0.6 | 86.8 +/- 0.6 | 75.7 +/- 1.6 | 83.9 |
| Execution-Guided | 70.8 +/- 0.4 | 51.8 +/- 0.9 | 86.3 +/- 0.8 | 75.4 +/- 1.6 | 71.1 |
| Neg. Perplexity | 73.3 +/- 0.5 | 55.0 +/- 1.2 | 70.4 +/- 2.0 | 65.1 +/- 2.8 | 66.0 |
| MBR Consensus | 71.6 +/- 0.5 | 51.9 +/- 0.7 | 72.5 +/- 1.2 | 62.3 +/- 2.2 | 64.6 |
| Majority Vote (normalized, compile only) | 71.2 +/- 0.4 | 51.5 +/- 0.6 | 72.2 +/- 1.4 | 61.5 +/- 2.5 | 64.1 |
| Majority Vote (exact, compile only) | 71.0 +/- 0.5 | 51.5 +/- 1.0 | 72.2 +/- 1.7 | 61.6 +/- 2.6 | 64.1 |
| Trigram Similarity | 66.2 +/- 1.3 | 51.6 +/- 0.9 | 69.3 +/- 0.5 | 60.9 +/- 3.7 | 62.0 |
| Random | 68.2 +/- 1.2 | 47.0 +/- 1.2 | 69.5 +/- 1.4 | 59.4 +/- 1.9 | 61.0 |
| UniXcoder (fine-tuned) | 66.8 +/- 1.2 | 46.4 +/- 0.7 | 68.0 +/- 1.2 | 59.5 +/- 3.2 | 60.2 |
| Bigram Similarity | 65.7 +/- 1.3 | 44.9 +/- 1.1 | 69.0 +/- 0.9 | 60.6 +/- 3.4 | 60.0 |
| RepE Correctness | 66.0 +/- 2.3 | 45.3 +/- 1.6 | 68.6 +/- 1.1 | 58.8 +/- 4.2 | 59.7 |
| Ensemble (text) | 66.5 +/- 1.6 | 42.2 +/- 0.7 | 68.9 +/- 0.3 | 60.8 +/- 3.5 | 59.6 |
| Edit Similarity | 64.8 +/- 1.9 | 43.0 +/- 2.5 | 69.4 +/- 1.2 | 61.4 +/- 2.6 | 59.6 |
| Identifier Similarity | 60.2 +/- 0.9 | 46.8 +/- 1.4 | 68.5 +/- 1.6 | 61.6 +/- 4.2 | 59.3 |
| Token F1 | 64.2 +/- 0.8 | 42.6 +/- 1.2 | 67.9 +/- 1.5 | 60.3 +/- 3.1 | 58.8 |
| AST Node Similarity | 61.0 +/- 1.2 | 39.9 +/- 1.5 | 67.9 +/- 0.6 | 58.7 +/- 3.3 | 56.9 |
| UniXcoder (base) | 62.1 +/- 1.6 | 38.5 +/- 0.9 | 67.5 +/- 1.5 | 59.3 +/- 3.5 | 56.9 |
| Control Flow Similarity | 58.4 +/- 2.3 | 35.9 +/- 0.7 | 67.9 +/- 1.4 | 61.5 +/- 2.5 | 55.9 |
| Majority Vote (normalized, all) | 65.2 +/- 0.5 | 48.0 +/- 0.3 | 37.4 +/- 0.6 | 15.0 +/- 0.2 | 41.4 |
| Majority Vote (exact, all) | 65.0 +/- 0.4 | 47.8 +/- 0.5 | 37.4 +/- 0.8 | 14.9 +/- 0.2 | 41.3 |

### Related Work Results

These values are taken from their respective publications and are not reruns
inside this pipeline.

| Method | C1 | C2 | C3 | C4 | Avg. |
|---|---:|---:|---:|---:|---:|
| SK2 Decompile | 69.0 | -- | -- | -- | 69.0 |
| CodeInverter | -- | 62.1 | -- | -- | 62.1 |
| ReF Decompile | -- | 61.4 | -- | -- | 61.4 |
| HELIOS | 55.9 | -- | -- | -- | 55.9 |
| SccDec + FAE | -- | 52.4 | -- | -- | 52.4 |
| SALT4Decompile | -- | 58.7 | -- | 26.4 | 42.5 |

## Key Findings

1. **Execution-guided reranking is the strongest advanced selector overall.**
   It reaches 86.3% on C3 and 75.4% on C4, within 0.5 and 0.3 percentage
   points of the corresponding oracle results.
2. **Negative perplexity is the strongest non-executing selector on average.**
   It obtains 66.0% average pass@1 and leads the static methods on C1, C2,
   and C4.
3. **MBR provides a useful consensus signal but is configuration-dependent.**
   It exceeds negative perplexity on C3, but not on the other configurations.
   The results alone do not establish likelihood collapse as the cause.
4. **RepE did not improve selection in this setup.** Its 59.7% average is
   below random selection and the stronger static selectors.
5. **Voting over all candidates is unreliable.** Non-compiling repeated
   candidates can dominate the vote, especially for ExeBench. Restricting
   voting and reranking to the compiling pool is substantially safer.

## Relevant Files

```text
RQ2_candidate_reranking/
|- README.md
|- selection_pipeline.py                  # Older 200-candidate HumanEval pipeline
|- selection_pipeline_mbr.py              # 100-candidate HumanEval MBR pipeline
|- selection_pipeline_execution_guided.py # HumanEval execution-guided pipeline
|- compute_execution_guided_cache.py       # HumanEval execution-guided cache builder
|- evaluate_all_metrics_execution_guided.py # Standalone execution-guided evaluator
|- evaluate_selection.py                  # Final evaluation from existing phase caches
|- compute_repe_correctness_cache.py       # RepE cache builder for ExeBench-style samples
|- compute_repe_correctness_cache_humaneval.py # HumanEval RepE cache builder
|- train_contrastive_unixcoder.py          # Optional UniXcoder fine-tuning
|- train_execution_guided_reranker.py      # Experimental learned execution proxy
|- unixcoder.py                            # UniXcoder implementation
|- beam_search_baselines.py                # vLLM beam/decoding baseline
|- beam_search_exebench.py                 # ExeBench beam-search baseline
|- run-sandboxed.sh                        # Candidate execution isolation
|- export_to_tex.py                        # LaTeX result formatting
`- selection_results.json                 # Preserved result artifact
```

The old `selection_pipeline.py` uses a 200-candidate configuration and should
not be used to reproduce the reported 100-candidate results. The learned
execution proxy in `train_execution_guided_reranker.py` was a later prototype
and is not part of the reported C1-C4 selector table.

## Generated Artifacts

The repository primarily tracks experiment source code. Candidate pools,
intermediate caches, checkpoints, and result files are created when the scripts
are executed and may be stored outside version control.

```text
cache/
|- candidates_<samples>_100.pkl
|- phase1_compilation.pkl
|- phase2_text_metrics.pkl
|- phase2_5_mbr.pkl
|- phase3_unixcoder.pkl
|- phase3_5_execution_guided.pkl
|- phase3_perplexity.pkl
|- phase3_7_repe_correctness.pkl
|- phase4_test_results.pkl
|- selection_results_*.json
`- selection_evaluation_results.json

checkpoints/
|- contrastive_unixcoder_best.pt
|- execution_guided_reranker.pt
`- other training checkpoints
```

Each of C1, C2, C3, and C4 should use a separate cache directory. These files
are outputs of the experiment rather than source files required in the Git
repository listing.

The current source listing still does not contain the comprehensive ExeBench
100-candidate reranking driver previously used to create all C3/C4 phase
caches. `beam_search_exebench.py` only covers the ExeBench beam-search baseline.
That advanced-reranking driver should be restored or consolidated into a final
configuration-driven pipeline before claiming one-command C1-C4 reproduction.

## Reproducibility Notes

- Use a separate cache directory for each configuration. Cache keys only use
  `(sample.idx, candidate_index)` and cannot detect a mismatched dataset/model.
- C3 and C4 candidate executables must be run in the sandbox.
- `oracle` and complete test outcomes are evaluation-only information and must
  not be used by deployable selectors.
- Execution-guided reranking requires a runnable partial harness and is not a
  static, execution-free selector.
- RepE requires labeled fit and validation samples before it can score unseen
  candidates without executing them.
- Generated candidates, caches, checkpoints, and JSON results are runtime
  artifacts and are not expected to appear in the source-file listing.
- A final unified pipeline should expose C1-C4 as explicit configurations
  rather than relying on manual source edits.
