# IR Course Project Proposal: LLM Routing for TREC Million LLMs Track

## Step 1: Choose a Task
**Task**: LLM Routing and Expertise Ranking
**Benchmark**: [TREC Million LLMs Track](https://trec-mllm.github.io)
**Description**: As organizations deploy ensembles of specialized LLMs, it becomes critical to identify which model is best suited for a specific user query. Given a query and a large set of LLM IDs, the task is to rank the models by their predicted expertise—likelihood of providing a high-quality answer—*without* querying the models at test time.

## Step 2: Read the Baseline
**Problem Statement**: Effectively route queries to the most capable LLM using only precomputed discovery data (previous model responses, capabilities, metadata) instead of real-time sampling.
**Dataset**: TREC Million LLMs Discovery Set (training queries, precomputed responses, metadata) and Development Set (dev queries and expertise labels/qrels).
**Baseline Method**: A standard Dense Retrieval baseline. For each LLM, we can aggregate its precomputed responses from the Discovery set and embed them using a standard sentence transformer (e.g., `all-MiniLM-L6-v2`). Given a test query, we embed the query and rank LLMs using cosine similarity between the query embedding and the LLM's aggregated response embeddings.
**Evaluation Metric**: Ranking metrics such as **NDCG@10**, **MRR** (Mean Reciprocal Rank), and **Recall@K**, since the ground truth is provided as a qrel of query-LLM pairs with expertise labels.

## Step 3: Set Up the Baseline
1. **Download the Dataset**: Access the Discovery and Development sets from the TREC Million LLMs track.
2. **Data Preprocessing**: Aggregate and clean the precomputed responses and metadata for each LLM.
3. **Implement Baseline**: Use `sentence-transformers` to generate embeddings for the query and the LLMs' historical responses. Compute cosine similarity scores to rank the LLM IDs.
4. **Validation**: Evaluate the baseline ranking against the Development set qrels to establish our initial NDCG@10 and MRR scores.

## Step 4: Propose an Improvement
**Proposed Method: Two-Stage Metadata-Aware Router (Hybrid Retrieval + Reranking)**
The baseline relies purely on semantic similarity of past responses, ignoring valuable metadata (e.g., model size, domain specialization, response length, language capabilities). We propose a two-stage approach:

1. **Stage 1 (Recall)**: Use a Hybrid Retrieval approach (**BM25 + Dense Retrieval**) on the LLM's historical responses to retrieve a top-K list of candidate LLMs.
2. **Stage 2 (Reranking)**: Train a lightweight reranker (e.g., an XGBoost classifier or a small Cross-Encoder model) that takes the Stage 1 score and concatenates it with the **LLM metadata features** (domain, parameter count, typical response style). The model will be trained on the Discovery/Dev set to predict the final expertise score.

**Why it's better**: This method leverages both lexical/semantic history (what the model has answered before) and explicit metadata (what the model is designed to do), addressing the shortcomings of a pure dense retrieval baseline.

## Step 5: Run Experiments
- **Setup**: Train the reranker on the Discovery set; validate and tune hyperparameters on a split of the Development set.
- **Comparison**: Compare the baseline (Dense Retrieval) against our proposed Two-Stage Router.
- **Evaluation**: Use the exact same evaluation metrics (NDCG@10, MRR) and the held-out Test set (or Development set if Test is not yet released).

## Step 6: Analyze Results
Our final report will include:
- **Performance Table**: Comparing Baseline vs. Proposed method across NDCG@10, MRR, and Recall@100.
- **Ablation Study**: Showing the contribution of BM25 vs Dense Retrieval vs Metadata Reranking.
- **Graphs**: Query-type analysis (e.g., does the reranker perform better on coding queries vs. medical queries?).
- **Insights & Limitations**: Discussion on why the method improved the results, scenarios where the router fails to select the best LLM, and potential future work (like using a smaller LLM as a dedicated router).

## Deliverables Checklist
- [ ] Approved project proposal (This document)
- [ ] Source code with instructions (`README.md`, `requirements.txt`, reproducible scripts)
- [ ] Final project report (PDF)
- [ ] Presentation slides
