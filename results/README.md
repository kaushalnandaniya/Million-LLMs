# Evaluation Results

This folder contains evaluation logs and summary outputs produced by the baseline and the improved router pipelines.

## Benchmark Performance Comparison

Evaluation was performed on the official TREC Development set (`qrels`) using standard Information Retrieval metrics (`ir_measures`):

| Model / Strategy | Architecture | nDCG@10 | nDCG@20 | MRR | Gain (nDCG@10) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Official TREC Baseline** | Dense Bi-Encoder (`all-MiniLM-L6-v2`) | 0.1604 | 0.1941 | 0.2269 | Baseline |
| **Initial L2R Router** | Hybrid (BM25 + BGE Base + BGE Large CE [560M] + KNN LogProbs) | 0.3280 | 0.3720 | 0.3840 | +104.5% |
| **Gemma-2B Hybrid L2R Router** | Granular Chunking + Gemma-2B Causal CE + 20 Domain Features + Residual Blend | **0.3409** | **0.3697** | **0.3998** | **+112.5%** |

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

