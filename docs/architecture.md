# System Architecture & Feature Engineering

## 1. Overview
The goal of this system is **zero-inference LLM routing**: given an input prompt $q$ and a pool of $N$ candidate LLMs $\mathcal{M} = \{m_1, \dots, m_N\}$ ($N = 150$), output a ranked permutation $\tau$ of $\mathcal{M}$ maximizing retrieval quality without querying any candidate model at test time.

```
                                [ User Query q ]
                                       |
       +-------------------------------+-------------------------------+
       |                               |                               |
[ 1. Category Topic Semantics ]  [ 2. Hybrid Profile Retrieval ]  [ 3. Deep Cross-Attention ]
  BGE-Large-en-v1.5 &             Dense Bi-Encoder (BGE/E5) +       BAAI/bge-reranker-v2-gemma
  e5-large-v2 vs Category Taxonomy  Lexical BM25 Over Profiles       (Causal Cross-Attention Logits)
       |                               |                               |
       +-------------------------------+-------------------------------+
                                       |
                    [ 4. Query-Aware Uncertainty Profiling ]
                     k-NN Search on Discovery Query Space
                     -> Extracts LLM's historical Mean LogProb,
                        Min LogProb, and Response Length on similar queries
                                       |
                    [ 5. Feature Engineering (23 Multi-Modal Features) ]
                     Includes Global LLM Priors and Bayesian Deviations
                                       |
                    [ 6. Learning-to-Rank Engine (XGBRanker) ]
                     Optuna Bayesian-Optimized with 5-Fold GroupKFold
                                       |
                             [ Ranked LLM Candidates ]
```

---

## 2. Feature Vector Design (23 Multi-Modal Features)

For each query-LLM pair $(q, m_i)$, the feature extractor constructs a 23-dimensional vector across five orthogonal retrieval and profiling modalities:

### A. Category-Aware Topic Semantics (Latent Taxonomy)
In the TREC Million LLMs benchmark, the 150 candidate models represent domain-specialized experts whose model IDs encode their fine-tuning Wikipedia category.
* `name_sim_bge`: Cosine similarity between query $q$ and clean topic name using `BAAI/bge-large-en-v1.5`.
* `name_sim_e5`: Asymmetric passage-query similarity with `intfloat/e5-large-v2`.
* `name_bm25`: BM25 lexical token matching against clean category topic tokens.

### B. Dense & Lexical Profile Matching
* `bge_sim`: Cosine similarity between $q$ and model's historical response corpus using `BAAI/bge-base-en-v1.5`.
* `bge_large_sim`: Similarity using `BAAI/bge-large-en-v1.5`.
* `e5_sim`: Asymmetric retrieval similarity with `intfloat/e5-large-v2`.
* `bm25_score`: Okapi BM25 score of tokenized query against model's response corpus.

### C. Deep Causal Cross-Encoder Reranking
* `ce_score`: Causal cross-attention logits from the 2-Billion parameter `BAAI/bge-reranker-v2-gemma` evaluated on the most relevant semantic response chunk.

### D. Query-Conditioned Uncertainty Profiling
* `avg_mean_lp`: Weighted average mean token log-probability across $k=5$ nearest historical discovery queries.
* `avg_min_lp`: Weighted average minimum token log-probability (worst-case token uncertainty).
* `avg_length`: Expected response length on similar historical queries.
* `global_mean_lp`: Model-specific global average token log-probability prior.
* `global_min_lp`: Model-specific baseline minimum token log-probability.
* `global_len`: Baseline expected response verbosity.
* `mean_knn_sim`: Mean cosine similarity of top-$k$ nearest discovery queries to $q$.
* `q_len`: User query word count.
* `diff_mean_lp`: `avg_mean_lp - global_mean_lp` (**Domain Specialization Spike**).
* `diff_length`: `avg_length - global_len` (Verbosity deviation).

### E. Domain Intent Indicators
* `is_code`: Boolean heuristic indicator for software engineering, syntax, and programming prompts.
* `is_math`: Quantitative, algebraic, and mathematical reasoning cues.
* `is_factual`: Factual lookup cues (e.g., "who was", "capital of", "when did").
* `is_creative`: Creative writing, essay, and storytelling cues.
* `is_reasoning`: Explanatory and causal reasoning cues ("why", "explain mechanism").

---

## 3. Learning-to-Rank Engine & Continuous Residual Blending

Instead of treating routing as point-wise regression or naive score averaging, we employ an **XGBRanker** utilizing listwise optimization (`objective='rank:ndcg'`).

### Continuous Residual Logit Blending
Tree-based models partition candidates using axis-aligned orthogonal splits, often creating discrete prediction ties at leaf nodes. To resolve ranking ambiguity at top ranks, continuous causal cross-encoder logits and category semantic similarities are blended as residual offsets:

$$\text{Final Score}(q, m_i) = \hat{y}_{\text{xgb}} + \lambda_{\text{ce}} \cdot \tilde{s}_{\text{ce}}(q, m_i) + \lambda_{\text{name}} \cdot \tilde{s}_{\text{name}}(q, m_i)$$

where scores are $z$-score normalized within query evaluation slices.

---

## 4. Benchmark Results

| Architecture | nDCG@10 | nDCG@20 | MRR | MAP |
| :--- | :---: | :---: | :---: | :---: |
| Official Benchmark Baseline | 0.1604 | 0.1873 | 0.2269 | 0.1250 |
| Category Topic Semantics Standalone | 0.2919 | 0.3201 | 0.3464 | 0.2185 |
| XGBRanker (20 Features) | 0.3365 | 0.3667 | 0.3966 | 0.2741 |
| Topic-Aware Gemma-2B SOTA Router | **0.3990** | **0.4217** | **0.4327** | **0.3457** |
