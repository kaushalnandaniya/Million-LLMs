import os
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

def train_and_save():
    results_path = 'results/results_enhanced.json'
    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found.")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    best_params = data['best_params']
    ce_blend = best_params.get('ce_blend', 0.0)
    name_blend = best_params.get('name_blend', 0.0)
    xgb_params = {k: v for k, v in best_params.items() if k not in ['ce_blend', 'name_blend']}
    xgb_params['objective'] = 'rank:ndcg'
    xgb_params['random_state'] = 42
    xgb_params['n_jobs'] = -1

    with open('data/dev_features_enhanced.pkl', 'rb') as f:
        df = pickle.load(f)

    feature_cols = [
        'name_sim_bge', 'name_sim_e5', 'name_bm25',
        'bge_sim', 'bge_large_sim', 'e5_sim',
        'bm25_score', 'ce_score',
        'avg_mean_lp', 'avg_min_lp', 'avg_length',
        'global_mean_lp', 'global_min_lp', 'global_len',
        'mean_knn_sim', 'q_len',
        'diff_mean_lp', 'diff_length',
        'is_code', 'is_math', 'is_factual', 'is_creative', 'is_reasoning'
    ]

    print("Fitting production XGBRanker on full dev dataset...")
    q_codes = df['query_id'].astype('category').cat.codes.values
    sort_idx = np.argsort(q_codes)

    X = df[feature_cols].values[sort_idx]
    y = df['label'].values[sort_idx]
    qid = q_codes[sort_idx]

    model = xgb.XGBRanker(**xgb_params)
    model.fit(X, y, qid=qid)

    os.makedirs('models', exist_ok=True)
    model_bundle = {
        'model': model,
        'feature_cols': feature_cols,
        'ce_blend': ce_blend,
        'name_blend': name_blend,
        'best_params': best_params
    }

    bundle_path = 'models/xgb_enhanced_router.pkl'
    with open(bundle_path, 'wb') as f:
        pickle.dump(model_bundle, f)
    print(f"Saved production router bundle to {bundle_path}!")

if __name__ == '__main__':
    train_and_save()
