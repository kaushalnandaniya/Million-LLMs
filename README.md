# Intelligent Query Routing Across LLM Ensembles (TREC Million LLMs Track)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-L2R-red.svg)](https://xgboost.readthedocs.io/)
[![Benchmark](https://img.shields.io/badge/Benchmark-TREC_Million_LLMs-green.svg)](https://trec-mllm.github.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An Information Retrieval (IR) framework for **zero-inference LLM routing**, designed and evaluated on the official **NIST TREC Million LLMs Track** benchmark.

Given an incoming user query and a large pool of candidate Large Language Models ($N = 150$), our system predicts and ranks the most capable models **without querying any LLM at test time**. By formulating model selection as a **Two-Stage Hybrid Learning-to-Rank (L2R)** problem, this router improves ranking performance from **0.1941 to 0.3720 nDCG@20 (+91.7% relative gain)** over the official dense retrieval baseline.

---

## Key Highlights

* **Zero-Inference Constraint:** Routing decisions execute in **< 15 milliseconds** on precomputed metadata, bypassing expensive multi-model test-time inference.
* **Hybrid Retrieval (Dense + Sparse):** Combines state-of-the-art dense semantic representations (`BAAI/bge-base-en-v1.5`) with exact lexical matching (`Rank-BM25`).
* **Query-Conditioned Uncertainty Profiling:** Performs $k$-NN search over the historical discovery query space to weight model confidence (token log-probabilities) on semantically similar tasks.
* **Bayesian Deviation Features:** Compares conditional task confidence against global LLM priors to identify specialized model competence spikes.
* **Deep Cross-Encoder Attention:** Precomputes deep query-profile cross-attention using large rerankers (`BAAI/bge-reranker-large` and 2B `bge-reranker-v2-gemma`).
* **Listwise Learning-to-Rank:** Trains an `XGBRanker` optimizing pairwise and listwise (`rank:ndcg`) objectives.
* **Zero-Leakage Validation:** Strictly evaluates via **5-Fold `GroupKFold`** partitioned on `query_id`, ensuring zero query leakage.

---

## Performance Benchmark

Evaluation conducted on the official TREC Development set (`qrels`) using standard IR metrics (`ir_measures`):

| Model / Approach | Retrieval Architecture | nDCG@10 | nDCG@20 | MRR | Relative Gain (nDCG@20) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Official TREC Baseline** | Dense Bi-Encoder (`all-MiniLM-L6-v2`) | 0.1604 | 0.1941 | 0.2269 | Baseline |
| **Two-Stage Hybrid L2R Router** | Hybrid (BM25 + BGE Dense + Cross-Encoder + KNN Uncertainty) + XGBRanker | **0.3280** | **0.3720** | **0.3840** | **+91.7%** |

---

## System Architecture

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
├── src/
│   ├── __init__.py
│   ├── baseline.py                 # Dense Bi-Encoder baseline
│   ├── router.py                   # Two-Stage Hybrid L2R Router (BGE Large Cross-Encoder)
│   ├── router_advanced.py          # Multi-Embedding Fusion Router (BGE + BGE-Large + E5)
│   ├── router_gemma.py             # 2B Causal Reranker Router with checkpointing & caching
│   ├── tune_router.py              # Optuna hyperparameter optimization for router
│   ├── tune_advanced.py            # Optuna tuning for multi-embedding pipeline
│   ├── tune_gemma.py               # Optuna tuning for Gemma router
│   ├── evaluate.py                 # TREC baseline evaluator (nDCG@10, nDCG@20, MRR)
│   ├── evaluate_router.py          # Router out-of-fold evaluation script
│   ├── mock_data.py                # Synthetic mock data generator for testing
│   └── gemma_reranker.py           # Unit test for Gemma-2B logit extraction
├── tests/
│   ├── __init__.py
│   ├── test_ce.py                  # Cross-Encoder GPU loading check
│   ├── test_st.py                  # Sentence Transformer sanity check
│   ├── test_sfr.py                 # SFR Mistral embedding test
│   ├── test_transformers.py        # Hugging Face sequence classification check
│   └── test_transformers_sfr.py    # Dual GPU tensor allocation test
├── data/
│   └── README.md                   # Instructions for placing TREC dataset files
└── results/
    └── README.md                   # Evaluation results and metric definitions
```

---

## Getting Started

### 1. Environment Setup

Clone the repository and install dependencies:
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

*(Optional)* For fast testing without downloading the full 1.5GB dataset, generate mock data:
```bash
python src/mock_data.py
```

### 3. Running the Baseline
Run the dense bi-encoder baseline and evaluate performance:
```bash
python src/baseline.py
python src/evaluate.py
```

### 4. Running the Two-Stage Hybrid Router
Train and evaluate the proposed Learning-to-Rank router using 5-fold GroupKFold:
```bash
python src/router.py
python src/evaluate_router.py
```

### 5. Hyperparameter Optimization
To run Bayesian optimization over XGBRanker hyperparameters using Optuna:
```bash
python src/tune_router.py
```

---

## Presentation Slides

A high-level project ideation presentation in LaTeX Beamer (16:9 widescreen format with TikZ architectural diagrams) is available in the [`presentation/`](presentation/) folder:
* Source: [`presentation/slides.tex`](presentation/slides.tex)
* Guide: [`presentation/README.md`](presentation/README.md) (ready for 1-click import into Overleaf)

---

## References & Acknowledgments

* **Benchmark:** [NIST TREC Million LLMs Track](https://trec-mllm.github.io)
* **Metrics:** Evaluated using the official [`ir-measures`](https://github.com/terrierteam/ir_measures) library.
* **Models:** Embeddings and rerankers powered by Hugging Face `transformers` and `sentence-transformers`.
