# Evaluation Results

This folder contains evaluation summaries, ablation studies, and official TREC submission files produced across all routing architectures.

## Benchmark Performance Comparison

All models are strictly evaluated on the official NIST TREC Development set (`48,300` Query-LLM pairs across `322` queries) using 5-Fold `GroupKFold` Cross-Validation (partitioned on `query_id` to guarantee zero query leakage) with `ir_measures`:

| Model / Strategy | Architecture | nDCG@10 | nDCG@20 | MRR | MAP | Relative Gain (nDCG@10) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Official TREC Baseline** | Dense Bi-Encoder (`all-MiniLM-L6-v2`) | 0.1604 | 0.1873 | 0.2269 | 0.1250 | Baseline |
| **BGE-Base Dense** | Standalone Bi-Encoder | 0.1905 | 0.2210 | 0.2601 | 0.1432 | +18.8% |
| **Category Topic Semantics** | Zero-Shot Category Name Embedding | 0.2919 | 0.3201 | 0.3464 | 0.2185 | +82.0% |
| **Initial L2R Router** | Hybrid (BM25 + BGE Base + BGE Large CE [560M] + KNN LogProbs) | 0.3280 | 0.3720 | 0.3840 | 0.2650 | +104.5% |
| **Gemma-2B Hybrid L2R Router** | Gemma-2B Causal CE + 20 Domain Features + Residual Blend | 0.3409 | 0.3697 | 0.3998 | 0.2815 | +112.5% |
| **Topic-Aware Gemma-2B SOTA Router** | Category Semantics + Causal Gemma CE + Optuna Bayesian Tuning | **0.3990** | **0.4217** | **0.4327** | **0.3457** | **+148.8%** |

Detailed model-by-model ablation breakdowns and architectural takeaways are documented in [`results/ablation_summary.md`](ablation_summary.md).

## Metric Definitions
* **nDCG@k (Normalized Discounted Cumulative Gain at rank k):** Measures graded relevance ranking quality, giving highest weight to expert models (`label = 2`) ranked at top positions.
* **MRR (Mean Reciprocal Rank):** Evaluates the rank reciprocal of the first competent LLM.
* **MAP (Mean Average Precision):** Evaluates global precision across all relevant candidates.

## Reproducing Evaluation & Ablation
```bash
# Generate complete markdown ablation table
python src/ablation.py

# Run real-time query router CLI
python src/route_query.py "How do I implement quicksort in Python?"
```
