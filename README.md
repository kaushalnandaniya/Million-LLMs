# Intelligent Query Routing Across LLM Ensembles (TREC Million LLMs Track)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-L2R-red.svg)](https://xgboost.readthedocs.io/)
[![Benchmark](https://img.shields.io/badge/Benchmark-TREC_Million_LLMs-green.svg)](https://trec-mllm.github.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An Information Retrieval (IR) framework for **zero-inference LLM routing**, designed and evaluated on the official **NIST TREC Million LLMs Track** benchmark.

Given an incoming user query and a large pool of candidate Large Language Models ($N = 150$), our system predicts and ranks the most capable models **without querying any LLM at test time**. By formulating model selection as a **Topic-Aware Hybrid Learning-to-Rank (L2R)** problem, this router improves ranking performance from **0.1604 to 0.3990 nDCG@10 (+148.8% relative gain)** and **0.1873 to 0.4217 nDCG@20 (+125.1% relative gain)** over the official NIST benchmark baseline.

---

## Key Highlights

* **Zero-Inference Constraint:** Routing decisions execute in **< 15 milliseconds** on precomputed metadata, bypassing expensive multi-model test-time inference.
* **Category-Aware Topic Semantics (+82.0% standalone gain):** Unlocks the latent Wikipedia category taxonomy of candidate LLM models using dual neural encoders (`BAAI/bge-large-en-v1.5` and `intfloat/e5-large-v2`).
* **Deep Causal Cross-Attention:** Granular response chunking scored via the 2-Billion parameter `BAAI/bge-reranker-v2-gemma` causal cross-encoder.
* **Query-Conditioned Uncertainty Profiling:** Performs $k$-NN search over the historical discovery query space to weight model confidence (token log-probabilities) on semantically similar tasks.
* **Bayesian Deviation Features:** Compares conditional task confidence against global LLM priors to identify specialized model competence spikes.
* **Listwise Learning-to-Rank Engine:** Trains an `XGBRanker` optimizing pairwise and listwise (`rank:ndcg`) objectives with continuous cross-attention residual blending.
* **Zero-Leakage Validation:** Strictly evaluates via **5-Fold `GroupKFold`** partitioned on `query_id`, ensuring zero query leakage across 48,300 Query-LLM pairs.
* **Interactive CLI:** Real-time terminal interface for testing arbitrary natural language queries against the 150 LLM candidates.

---

## Performance Benchmark

All models are strictly evaluated on the official TREC Development set (`48,300` Query-LLM evaluations across `322` queries) using standard IR metrics (`ir_measures`):

| Model / Approach | Retrieval Architecture | nDCG@10 | nDCG@20 | MRR | MAP | Relative Gain (nDCG@10) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Official TREC Baseline** | Dense Bi-Encoder (`all-MiniLM-L6-v2`) | 0.1604 | 0.1873 | 0.2269 | 0.1250 | Baseline |
| **BGE-Base Dense** | Standalone Bi-Encoder | 0.1905 | 0.2210 | 0.2601 | 0.1432 | +18.8% |
| **Category Topic Semantics** | Zero-Shot Category Name Embedding | 0.2919 | 0.3201 | 0.3464 | 0.2185 | +82.0% |
| **Initial L2R Router** | Hybrid (BM25 + BGE Base + BGE Large CE [560M] + KNN LogProbs) | 0.3280 | 0.3720 | 0.3840 | 0.2650 | +104.5% |
| **Gemma-2B Hybrid L2R Router** | Gemma-2B Causal CE + 20 Domain Features + Residual Blend | 0.3409 | 0.3697 | 0.3998 | 0.2815 | +112.5% |
| **Topic-Aware Gemma-2B Router (SOTA)** | Category Semantics + Causal Gemma CE + Optuna Bayesian Tuning | **0.3990** | **0.4217** | **0.4327** | **0.3457** | **+148.8%** |

Full model-by-model ablation tables and architectural insights are documented in [`results/ablation_summary.md`](results/ablation_summary.md).

---

## Interactive Query Router CLI

Test any prompt interactively or via single-command routing:

```bash
# Example 1: Coding query
python src/route_query.py "How do I write a fast sorting algorithm in Python?"
```

```text
===========================================================================
QUERY: How do I write a fast sorting algorithm in Python?
DETECTED INTENT DOMAIN: Coding, Creative Writing, Reasoning/Explanatory
===========================================================================
Rank  Recommended Candidate LLM           Score         Confidence
---------------------------------------------------------------------------
1     Mathematics                         3.5358        11.3%
2     Writing                             2.7760        5.3%
3     Applied_mathematics                 2.6603        4.7%
4     Fields_of_mathematics               2.4278        3.7%
5     Computing                           2.4111        3.7%
===========================================================================
```

```bash
# Example 2: Geography & History query
python src/route_query.py "What is the capital of Australia and who founded it?"
```

```text
===========================================================================
QUERY: What is the capital of Australia and who founded it?
DETECTED INTENT DOMAIN: General Knowledge
===========================================================================
Rank  Recommended Candidate LLM           Score         Confidence
---------------------------------------------------------------------------
1     Geography                           3.2172        7.5%
2     Geography_by_location               3.0198        6.2%
3     Geography_by_country                2.7481        4.7%
4     History_by_location                 2.7473        4.7%
5     Geography_by_period                 2.6711        4.4%
===========================================================================
```

---

## System Architecture

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

Detailed feature descriptions and mathematical formulations are documented in [`docs/architecture.md`](docs/architecture.md).

---

## Repository Structure

```
Million-LLMs/
├── .gitignore                      # Git exclusion rules for large datasets & checkpoints
├── README.md                       # Main repository documentation
├── requirements.txt                # Python package dependencies
├── docs/
│   ├── project_proposal.md         # Formal course project proposal
│   └── architecture.md             # Feature vector engineering and mathematical details
├── presentation/
│   ├── slides.tex                  # LaTeX Beamer presentation (16:9 widescreen)
│   └── README.md                   # Slide compilation guide
├── models/
│   └── README.md                   # Model training and artifact serialization guide
├── src/
│   ├── __init__.py
│   ├── route_query.py              # Interactive & command-line zero-inference query router
│   ├── ablation.py                 # Academic benchmark & ablation report generator
│   ├── tune_enhanced.py            # Optuna Bayesian tuning for Category-Aware SOTA router
│   ├── train_and_save_final.py     # Fits and exports production XGBRanker model bundle
│   ├── cache_router_assets.py      # Precomputes offline embeddings for 0.2s instant boot
│   ├── baseline.py                 # Dense Bi-Encoder baseline
│   ├── router.py                   # Two-Stage Hybrid L2R Router (BGE Large Cross-Encoder)
│   ├── router_gemma.py             # 2B Causal Reranker Router with checkpointing & caching
│   ├── tune_gemma.py               # Optuna tuning for Gemma router
│   ├── evaluate.py                 # TREC baseline evaluator (nDCG@10, nDCG@20, MRR)
│   ├── evaluate_router.py          # Router out-of-fold evaluation script
│   └── mock_data.py                # Synthetic mock data generator for testing
├── tests/
│   ├── __init__.py
│   ├── test_ce.py                  # Cross-Encoder GPU loading check
│   ├── test_st.py                  # Sentence Transformer sanity check
│   └── test_transformers.py        # Hugging Face sequence classification check
├── data/
│   └── README.md                   # Instructions for placing TREC dataset files
└── results/
    ├── README.md                   # Evaluation results overview
    ├── ablation_summary.md         # Full markdown ablation table
    ├── results_enhanced.json       # Best SOTA hyperparameter and metric logs
    └── results_gemma.json          # Gemma-2B baseline logs
```

---

## Getting Started

### 1. Environment Setup

```bash
git clone https://github.com/kaushalnandaniya/Million-LLMs.git
cd Million-LLMs

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the dataset from the [TREC Million LLMs Track](https://trec-mllm.github.io) into `data/data/`:
* `llm_discovery_metadata.json`
* `llm_discovery_data.json`
* `llm_dev_data.tsv`
* `llm_dev_qrels.txt`

### 3. Running the SOTA Router
```bash
# 1. Run the interactive router
python src/route_query.py "Explain the difference between TCP and UDP protocols"

# 2. View full ablation study
python src/ablation.py
```

### 4. Hyperparameter Optimization & Training
```bash
# Run Optuna Bayesian optimization
python src/tune_enhanced.py

# Fit and export final production model bundle
python src/train_and_save_final.py
```

---

## References & Acknowledgments

* **Benchmark:** [NIST TREC Million LLMs Track (2025)](https://trec-mllm.github.io)
* **Metrics:** Evaluated using the official [`ir-measures`](https://github.com/terrierteam/ir_measures) library.
* **Models:** Embeddings and rerankers powered by Hugging Face `transformers` and `sentence-transformers`.
