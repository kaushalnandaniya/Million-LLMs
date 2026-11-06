# Evaluation Results

This folder contains evaluation logs and summary outputs produced by the baseline and the improved router pipelines.

## Benchmark Performance Comparison

Evaluation was performed on the official TREC Development set (`qrels`) using standard Information Retrieval metrics (`ir_measures`):

| Model / Strategy | Architecture | nDCG@10 | nDCG@20 | MRR | Relative Gain (nDCG@20) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Official TREC Baseline** | Dense Bi-Encoder (`all-MiniLM-L6-v2`) | 0.1604 | 0.1941 | 0.2269 | — |
| **Two-Stage L2R Router** | Hybrid (BM25 + BGE Dense + Cross-Encoder + KNN Uncertainty) + XGBRanker | **0.3280** | **0.3720** | **0.3840** | **+91.7%** |

## Metric Definitions
* **nDCG@k (Normalized Discounted Cumulative Gain at rank k):** Measures graded relevance ranking quality, giving higher weight to placing top-expertise models at top ranks.
* **MRR (Mean Reciprocal Rank):** Evaluates the rank position of the first highly competent LLM.

## Reproducing Evaluation
To compute metrics from generated TREC run files:
```bash
# Evaluate baseline
python src/evaluate.py

# Evaluate improved router
python src/evaluate_router.py
```
