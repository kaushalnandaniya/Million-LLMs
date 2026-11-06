# Dataset Directory

This directory is designated for the official **TREC Million LLMs Track** dataset files.

## Dataset Structure
Place the extracted benchmark files into `data/data/`:
* `llm_discovery_metadata.json`: Historical queries, model responses, and token log-probabilities across 150 LLMs.
* `llm_discovery_data.json`: Precomputed model responses.
* `llm_dev_data.tsv`: Evaluation queries (`query_id \t text`).
* `llm_dev_qrels.txt`: Ground truth relevance judgements (`query_id 0 llm_id relevance_score`).

## Quick Mock Testing
If running in a lightweight environment without the full 1.5GB dataset, generate synthetic mock benchmark data using:
```bash
python src/mock_data.py
```
This generates miniature `data/mock_discovery_metadata.json`, `data/mock_dev_queries.tsv`, and `data/mock_qrels.txt` for fast end-to-end pipeline verification.

