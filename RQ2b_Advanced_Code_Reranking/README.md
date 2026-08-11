# RQ2: Candidate Reranking for LLM4Decompile

**RQ**: How effective are different selection/reranking strategies for improving pass@1 from pass@k decompilation results?

## Setup

- **Model**: LLM4Binary/llm4decompile-9b-v2
- **Dataset**: decompile-eval (656 samples)
- **Candidates**: 200 per sample via temperature sampling (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)
- **Evaluation**: Functional correctness (compilation + test passing), Levenshtein similarity, CodeBLEU

## Pipeline

```text
Step 0 (optional):  train_contrastive_unixcoder.py   -> checkpoints/contrastive_unixcoder_best.pt
Step 1:             selection_pipeline.py             -> cache/candidates_656_200.pkl
Step 2:             evaluate_all_metrics.py           -> cache/phase{1..4}_*.pkl
Step 2b (optional): selection_pipeline_execution_guided.py -> cache/phase3_5_execution_guided.pkl
Step 3:             evaluate_selection.py             -> selection_evaluation_results.json
Step 4:             export_to_tex.py                  -> LaTeX tables (stdout)
```

### Quick run (uses pre-computed cache)

```bash
cd v2/RQ2_candidate_reranking
python evaluate_selection.py
python export_to_tex.py
```

### Full regeneration

```bash
export CUDA_VISIBLE_DEVICES=2,3

# Step 0: Train contrastive UniXcoder (optional if checkpoint exists)
python train_contrastive_unixcoder.py

# Step 1: Generate 200 candidates per sample + compile check + text/AST metrics
python selection_pipeline.py

# Step 2: Compute MBR + UniXcoder metrics + run functional tests
python evaluate_all_metrics.py

# Step 2b: Compute execution-guided reranking scores on the same candidate pool
python selection_pipeline_execution_guided.py

# Step 3: Evaluate all selection strategies
python evaluate_selection.py

# Step 4: Export LaTeX tables
python export_to_tex.py
```

## Selection Strategies

| Category | Metric | Description |
|----------|--------|-------------|
| Baseline | random | Random compiling candidate |
| Baseline | oracle | Best possible (any passing candidate) |
| Text | edit_similarity | Normalized Levenshtein distance |
| Text | token_f1 | Token-level F1 overlap |
| Text | bigram_sim / trigram_sim | N-gram Jaccard similarity |
| AST | ast_node_sim | AST node type distribution cosine similarity |
| AST | control_flow_sim | Control flow construct count similarity |
| AST | identifier_sim | Identifier set Jaccard similarity |
| Consensus | mbr_consensus | Minimum Bayes Risk consensus over candidate set |
| Execution | execution_guided | Partial-test checkpoint score over the compiled candidate |
| Embedding | unixcoder_base | Pre-trained UniXcoder cosine similarity |
| Embedding | unixcoder_finetuned | Contrastive fine-tuned UniXcoder |
| Ensemble | ensemble_text | 0.4*bigram + 0.4*trigram + 0.2*token_f1 |
| Ensemble | ensemble_unixcoder_ft | 0.5*unixcoder_ft + 0.25*bigram + 0.25*trigram |

## Files

```text
RQ2_candidate_reranking/
|- README.md
|- unixcoder.py                    # Microsoft UniXcoder model (MIT license)
|- train_contrastive_unixcoder.py  # Step 0: Finetune UniXcoder (InfoNCE)
|- selection_pipeline.py           # Step 1: Generate + compile + text/AST metrics
|- evaluate_all_metrics.py         # Step 2: MBR + UniXcoder + tests
|- selection_pipeline_execution_guided.py  # Step 2b: Execution-guided reranking cache
|- evaluate_selection.py           # Step 3: Final pass@1 / Levenshtein / CodeBLEU
|- export_to_tex.py                # Step 4: LaTeX table output
`- cache/                          # Auto-created, holds intermediate .pkl files
```

## Results

Run the pipeline to regenerate the current MBR-focused results:

```bash
python evaluate_all_metrics.py
python evaluate_selection.py
python export_to_tex.py
```


| Selector | C1 | C2 | C3 | C4 | Avg. |
| :--- | :--- | :--- | :--- | :--- | :--- |
| *Oracle* | *92.7 ? 1.0* | *80.4 ? 0.6* | *86.8 ? 0.6* | *75.7 ? 1.6* | *83.9* |
| Execution-Guided | 70.8 ? 0.4 | 51.8 ? 0.9 | **86.3 ? 0.8** | **75.4 ? 1.6** | **71.1** |
| Neg. Perplexity | **73.3 ? 0.5** | 55.0 ? 1.2 | 70.4 ? 2.0 | 65.1 ? 2.8 | 66.0 |
| MBR Consensus | 71.6 ? 0.5 | **51.9 ? 0.7** | 72.5 ? 1.2 | 62.3 ? 2.2 | 64.6 |
| Majority Vote<sup>a</sup> | 71.2 ? 0.4 | 51.5 ? 0.6 | 72.2 ? 1.4 | 61.5 ? 2.5 | 64.1 |
| Majority Vote<sup>b</sup> | 71.0 ? 0.5 | 51.5 ? 1.0 | 72.2 ? 1.7 | 61.6 ? 2.6 | 64.1 |
| Trigram Similarity | 66.2 ? 1.3 | 51.6 ? 0.9 | 69.3 ? 0.5 | 60.9 ? 3.7 | 62.0 |
| Random | 68.2 ? 1.2 | 47.0 ? 1.2 | 69.5 ? 1.4 | 59.4 ? 1.9 | 61.0 |
| UniXcoder (Fine-tuned) | 66.8 ? 1.2 | 46.4 ? 0.7 | 68.0 ? 1.2 | 59.5 ? 3.2 | 60.2 |
| Bigram Similarity | 65.7 ? 1.3 | 44.9 ? 1.1 | 69.0 ? 0.9 | 60.6 ? 3.4 | 60.0 |
| RepE Correctness | 66.0 ? 2.3 | 45.3 ? 1.6 | 68.6 ? 1.1 | 58.8 ? 4.2 | 59.7 |
| Ensemble (Text) | 66.5 ? 1.6 | 42.2 ? 0.7 | 68.9 ? 0.3 | 60.8 ? 3.5 | 59.6 |
| Edit Similarity | 64.8 ? 1.9 | 43.0 ? 2.5 | 69.4 ? 1.2 | 61.4 ? 2.6 | 59.6 |
| Identifier Similarity | 60.2 ? 0.9 | 46.8 ? 1.4 | 68.5 ? 1.6 | 61.6 ? 4.2 | 59.3 |
| Token F1 | 64.2 ? 0.8 | 42.6 ? 1.2 | 67.9 ? 1.5 | 60.3 ? 3.1 | 58.8 |
| AST Node Similarity | 61.0 ? 1.2 | 39.9 ? 1.5 | 67.9 ? 0.6 | 58.7 ? 3.3 | 56.9 |
| UniXcoder (Base) | 62.1 ? 1.6 | 38.5 ? 0.9 | 67.5 ? 1.5 | 59.3 ? 3.5 | 56.9 |
| Control Flow Similarity | 58.4 ? 2.3 | 35.9 ? 0.7 | 67.9 ? 1.4 | 61.5 ? 2.5 | 55.9 |
| Majority Vote<sup>c</sup> | 65.2 ? 0.5 | 48.0 ? 0.3 | 37.4 ? 0.6 | 15.0 ? 0.2 | 41.4 |
| Majority Vote<sup>d</sup> | 65.0 ? 0.4 | 47.8 ? 0.5 | 37.4 ? 0.8 | 14.9 ? 0.2 | 41.3 |
| **Related Work** *(Scores taken from publication)* | | | | | |
| SK<sup>2</sup> Decompile | 69.0 | -- | -- | -- | 69.0 |
| CodeInverter | -- | 62.1 | -- | -- | 62.1 |
| ReF Decompile | -- | 61.4 | -- | -- | 61.4 |
| HELIOS | 55.9 | -- | -- | -- | 55.9 |
| SccDec + FAE | -- | 52.4 | -- | -- | 52.4 |
| SALT4Decompile | -- | 58.7 | -- | 26.4 | 42.5 |

<sup>a</sup> Normalized, Compile Only  
<sup>b</sup> Exact, Compile Only  
<sup>c</sup> Normalized, All  
<sup>d</sup> Exact, All



## Key Findings

1. **Execution-Guided is the Top Dynamic Selector:** Execution-guided reranking outperforms all other methods by a wide margin on ExeBench (86.3% on C3, 75.4% on C4), closing the gap to within 0.5 percentage points of the theoretical Oracle limit.
2. **Perplexity is the Best Static Selector:** Where execution harnesses aren't available, Negative Perplexity provides the strongest static selection signal (leading on C1, C2, and C4, averaging 66.0% overall pass@1).
3. **Consensus (MBR) Takes Over During Likelihood Collapse:** MBR Consensus only outperforms Perplexity on configuration C3 (72.5% vs 70.4%), which occurs because complex out-of-distribution code causes token likelihoods to flatten.
