import os
import pickle
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from router_gemma import extract_discovery_data, build_features

def main():
    data_dir = 'data/data'
    cache_path = 'data/dev_features_base.pkl'
    
    if os.path.exists(cache_path):
        print(f"Base feature cache already exists at {cache_path}!")
        return

    print("Loading queries and qrels...")
    dev_queries = []
    with open(f'{data_dir}/llm_dev_data.tsv', 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                dev_queries.append({'query_id': parts[0], 'text': parts[1]})
    dev_queries_df = pd.DataFrame(dev_queries)

    qrels_df = pd.read_csv(f'{data_dir}/llm_dev_qrels.txt', sep=' ', names=['query_id', 'zero', 'doc_id', 'relevance'], dtype={'query_id': str, 'doc_id': str})
    qrels_df_router = qrels_df.copy()
    qrels_df_router.rename(columns={'doc_id': 'llm_id', 'relevance': 'expertise_score'}, inplace=True)

    print("Extracting discovery metadata...")
    discovery_queries, discovery_stats, llm_global_stats, llm_profiles, llm_responses_raw = extract_discovery_data(f'{data_dir}/llm_discovery_metadata.json')

    print("Loading bi-encoder models...")
    models = {
        'bge': SentenceTransformer('BAAI/bge-base-en-v1.5'),
        'bge_large': SentenceTransformer('BAAI/bge-large-en-v1.5'),
        'e5': SentenceTransformer('intfloat/e5-large-v2')
    }

    print("Building base feature vectors...")
    X, y, groups = build_features(
        dev_queries_df,
        qrels_df_router,
        discovery_queries,
        discovery_stats,
        llm_global_stats,
        models,
        llm_profiles,
        llm_responses_raw=llm_responses_raw,
        k_nearest=5
    )

    feature_col_names = [
        'bge_sim', 'bge_large_sim', 'e5_sim',
        'bm25_score', 'ce_score',
        'avg_mean_lp', 'avg_min_lp', 'avg_length',
        'global_mean_lp', 'global_min_lp', 'global_len',
        'mean_knn_sim', 'q_len',
        'diff_mean_lp', 'diff_length',
        'is_code', 'is_math', 'is_factual', 'is_creative', 'is_reasoning'
    ]

    df = pd.DataFrame(X, columns=feature_col_names)
    df['query_id'] = groups
    df['llm_id'] = qrels_df_router['llm_id'].values
    df['label'] = y

    with open(cache_path, 'wb') as f:
        pickle.dump(df, f)
    print(f"Successfully saved {len(df)} feature rows to {cache_path}!")

if __name__ == '__main__':
    main()
