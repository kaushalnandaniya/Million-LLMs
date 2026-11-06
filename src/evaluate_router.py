import pandas as pd
import ir_measures
from ir_measures import *
import os

def evaluate_router():
    data_dir = 'data/data'
    if not os.path.exists(f'{data_dir}/llm_dev_qrels.txt') or not os.path.exists('results/router_run.trec'):
        print("Required files not found. Please run src/router.py first.")
        return

    # Load FULL ground truth qrels
    qrels_df = pd.read_csv(f'{data_dir}/llm_dev_qrels.txt', sep=' ', names=['query_id', 'zero', 'doc_id', 'relevance'], dtype={'query_id': str, 'doc_id': str})
    
    # Load run results (OOF predictions for full set)
    run_df = pd.read_csv('results/router_run.trec', sep=' ', names=['query_id', 'Q0', 'doc_id', 'rank', 'score', 'run_id'], dtype={'query_id': str, 'doc_id': str})
    
    metrics = [MRR, nDCG@10, nDCG@20]
    
    print("Evaluating XGBoost Router (OOF Predictions)...")
    results = ir_measures.calc_aggregate(metrics, qrels_df, run_df)
    
    print("\n--- Router Evaluation Results ---")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    evaluate_router()
