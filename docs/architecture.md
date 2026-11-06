# System Architecture & Feature Engineering

## 1. Overview
The goal of this system is **zero-inference LLM routing**: given an input prompt $q$ and a pool of $N$ candidate LLMs $\mathcal{M} = \{m_1, \dots, m_N\}$, output a ranked permutation $\tau$ of $\mathcal{M}$ maximizing retrieval quality without querying any model at inference time.

```
                         [ User Query q ]
                                |
      +-------------------------+-------------------------+
      |                         |                         |
[ 1. Dense Semantic ]   [ 2. Lexical BM25 ]   [ 3. Deep Cross-Encoder ]
 BAAI/bge-base-en-v1.5      Okapi BM25           BAAI/bge-reranker-large /
 (Embedding Similarity)   (Keyword Overlap)      bge-reranker-v2-gemma
      |                         |                         |
      +-------------------------+-------------------------+
                                |
             [ 4. Query-Aware Uncertainty Profiling ]
              k-NN Search on Discovery Query Space
              -> Extracts LLM's historical Mean LogProb,
                 Min LogProb, and Response Length on similar queries
                                |
             [ 5. Feature Vector Construction (13 Features) ]
              Includes global LLM priors and Delta-deviations
                                |
             [ 6. Learning-to-Rank Engine (XGBRanker) ]
              Tuned via Optuna with 5-Fold GroupKFold Validation
                                |
                      [ Ranked LLM Candidates ]
```

---

## 2. Feature Vector Design (13 Features)

For each query-LLM pair $(q, m_i)$, the feature extractor constructs a 13-dimensional vector:

| # | Feature Name | Description | Intuition / Signal |
|---|---|---|---|
| 1 | `sim_score` | Cosine similarity $\cos(\mathbf{e}_q, \mathbf{e}_{m_i})$ using `BAAI/bge-base-en-v1.5` | Dense semantic relevance between prompt and model's historical response profile. |
| 2 | `bm25_score` | Okapi BM25 score of tokenized query against model corpus | Captures exact lexical matches (critical for programming languages, math symbols, technical jargon). |
| 3 | `ce_score` | Cross-attention score from `BAAI/bge-reranker-large` / `bge-reranker-v2-gemma` | Deep token-level cross-attention between query and model expertise summary. |
| 4 | `avg_mean_lp` | Weighted average mean log-probability across $k=5$ nearest discovery queries | Local conditional confidence: how confident was model $m_i$ on similar questions in the past? |
| 5 | `avg_min_lp` | Weighted average minimum token log-probability across $k$-NN queries | Measures worst-case token uncertainty (detects hallucination vulnerability). |
| 6 | `avg_length` | Weighted average generated response length on $k$-NN queries | Expected verbosity/depth for this topic. |
| 7 | `global_mean_lp` | Global average mean log-probability of model $m_i$ across all historical prompts | Model-specific Bayesian prior for confidence. |
| 8 | `global_min_lp` | Global average minimum token log-probability | Model-specific Bayesian prior for minimum confidence. |
| 9 | `global_len` | Global average response length of model $m_i$ | Model-specific baseline verbosity. |
| 10 | `mean_knn_sim` | Average cosine similarity of the top-$k$ nearest discovery queries to $q$ | Query familiarity: how close is query $q$ to the historical discovery domain? |
| 11 | `q_len` | Word count of user query $q$ | Query complexity indicator. |
| 12 | `delta_mean_lp` | `avg_mean_lp - global_mean_lp` | **Domain Specialization Spike:** Is model $m_i$ significantly *more confident than its baseline* on this topic? |
| 13 | `delta_length` | `avg_length - global_len` | Response length anomaly for this query category. |

---

## 3. Learning-to-Rank Formulation

Instead of treating ranking as point-wise regression (which optimizes mean squared error uniformly across all items), we train an **XGBRanker** utilizing:
* **Pairwise Ranking Objective (`rank:pairwise`):** Optimizes the pairwise order between candidate models.
* **Listwise Ranking Objective (`rank:ndcg`):** Directly maximizes Discounted Cumulative Gain over the entire candidate list for each query.

### Zero-Leakage GroupKFold Validation
Because each query has 150 candidate LLM rows in the dataset (48,300 total rows across dev queries), standard random cross-validation would leak query semantics between train and test folds. 

We enforce **5-Fold `GroupKFold` grouped strictly on `query_id`**:
$$\text{Fold}_k \cap \text{Fold}_j = \emptyset \quad \forall \; q \in \mathcal{Q}$$
All query-LLM pairs for a given query are confined to either the training or the evaluation partition, ensuring that evaluation metrics reflect performance on truly unseen queries.

