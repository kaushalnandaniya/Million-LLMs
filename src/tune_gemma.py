import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import ir_measures
from ir_measures import *
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

from router_gemma import extract_discovery_data, build_features

def main():
    data_dir = 'data/data'
    
    dev_queries = []
    with open(f'{data_dir}/llm_dev_data.tsv', 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                dev_queries.append({'query_id': parts[0], 'text': parts[1]})
    dev_queries_df = pd.DataFrame(dev_queries)
    
    qrels_df = pd.read_csv(f'{data_dir}/llm_dev_qrels.txt', sep=' ', names=['query_id', 'zero', 'doc_id', 'relevance'], dtype={'query_id': str, 'doc_id': str})
    qrels_df_eval = qrels_df[['query_id', 'doc_id', 'relevance']].copy()
    
    qrels_df_router = qrels_df.copy()
    qrels_df_router.rename(columns={'doc_id': 'llm_id', 'relevance': 'expertise_score'}, inplace=True)
    
    discovery_queries, discovery_stats, llm_global_stats, llm_profiles, llm_responses_raw = extract_discovery_data(f'{data_dir}/llm_discovery_metadata.json')
    
    models = {
        'bge': SentenceTransformer('BAAI/bge-base-en-v1.5'),
        'bge_large': SentenceTransformer('BAAI/bge-large-en-v1.5'),
        'e5': SentenceTransformer('intfloat/e5-large-v2')
    }
    
    print("Extracting features globally for all trials...")
    X, y, groups = build_features(dev_queries_df, qrels_df_router, discovery_queries, discovery_stats, llm_global_stats, models, llm_profiles, llm_responses_raw=llm_responses_raw, k_nearest=5)
    
    group_encoder = LabelEncoder()
    encoded_groups = group_encoder.fit_transform(groups)
    
    def evaluate_model(param, ce_blend=0.0):
        gkf = GroupKFold(n_splits=5)
        oof_predictions = np.zeros(len(y))
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, y_train, groups_train = X[train_idx], y[train_idx], encoded_groups[train_idx]
            X_test, groups_test = X[test_idx], encoded_groups[test_idx]
            
            train_sort_idx = np.argsort(groups_train)
            X_train, y_train, groups_train = X_train[train_sort_idx], y_train[train_sort_idx], groups_train[train_sort_idx]
            
            test_sort_idx = np.argsort(groups_test)
            X_test, groups_test = X_test[test_sort_idx], groups_test[test_sort_idx]
            
            xgb_model = xgb.XGBRanker(**param)
            xgb_model.fit(X_train, y_train, qid=groups_train)
            
            preds_sorted = xgb_model.predict(X_test)
            preds_unsorted = np.zeros_like(preds_sorted)
            preds_unsorted[test_sort_idx] = preds_sorted
            
            if ce_blend > 0:
                ce_test = X[test_idx, 4]  # index 4 is ce_score
                preds_std = (preds_unsorted - np.mean(preds_unsorted)) / (np.std(preds_unsorted) + 1e-8)
                ce_std = (ce_test - np.mean(ce_test)) / (np.std(ce_test) + 1e-8)
                combined = preds_std + ce_blend * ce_std
                oof_predictions[test_idx] = combined
            else:
                oof_predictions[test_idx] = preds_unsorted
                
        return oof_predictions

    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'objective': 'rank:ndcg',
            'random_state': 42,
            'n_jobs': -1
        }
        ce_blend = trial.suggest_float('ce_blend', 0.0, 0.3)
        
        oof_predictions = evaluate_model(param, ce_blend)
        
        run_df = qrels_df_router.copy()
        run_df['score'] = oof_predictions
        run_df.rename(columns={'llm_id': 'doc_id'}, inplace=True)
        run_df_eval = run_df[['query_id', 'doc_id', 'score']].copy()
        
        metrics = [nDCG@20]
        results = ir_measures.calc_aggregate(metrics, qrels_df_eval, run_df_eval)
        
        return results[nDCG@20]

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("\n" + "="*50, flush=True)
    print("BEST TRIAL RESULTS:", flush=True)
    trial = study.best_trial
    print(f"  Best nDCG@20: {trial.value:.4f}", flush=True)
    print("  Best Hyperparameters: ", flush=True)
    for key, value in trial.params.items():
        print(f"    {key}: {value}", flush=True)
    print("="*50 + "\n", flush=True)

    # Full Evaluation on all official TREC metrics with best params
    best_ce_blend = trial.params.get('ce_blend', 0.0)
    best_xgb_params = {k: v for k, v in trial.params.items() if k != 'ce_blend'}
    best_xgb_params['objective'] = 'rank:ndcg'
    best_xgb_params['random_state'] = 42
    best_xgb_params['n_jobs'] = -1
    
    final_oof_preds = evaluate_model(best_xgb_params, best_ce_blend)
    
    final_run_df = qrels_df_router.copy()
    final_run_df['score'] = final_oof_preds
    final_run_df.rename(columns={'llm_id': 'doc_id'}, inplace=True)
    final_run_eval = final_run_df[['query_id', 'doc_id', 'score']].copy()
    
    full_metrics = [MRR, nDCG@10, nDCG@20]
    agg_results = ir_measures.calc_aggregate(full_metrics, qrels_df_eval, final_run_eval)
    
    print("--- Final Official Evaluation Results ---", flush=True)
    for metric, score in agg_results.items():
        print(f"  {metric}: {score:.4f}", flush=True)

    # Save official TREC run file
    os.makedirs('results', exist_ok=True)
    trec_records = []
    final_run_df_export = qrels_df_router.copy()
    final_run_df_export['predicted_score'] = final_oof_preds
    for qid, group in final_run_df_export.groupby('query_id'):
        group_sorted = group.sort_values(by='predicted_score', ascending=False)
        for rank, (_, row) in enumerate(group_sorted.iterrows()):
            trec_records.append({
                'query_id': qid,
                'Q0': 'Q0',
                'llm_id': row['llm_id'],
                'rank': rank + 1,
                'score': row['predicted_score'],
                'run_id': 'gemma_hybrid_l2r'
            })
    trec_df = pd.DataFrame(trec_records)
    trec_df.to_csv('results/gemma_router_run.trec', sep=' ', index=False, header=False)
    print("Saved official TREC run to results/gemma_router_run.trec", flush=True)

    import json
    results_data = {
        'best_ndcg_20': float(agg_results[nDCG@20]),
        'best_ndcg_10': float(agg_results[nDCG@10]),
        'best_mrr': float(agg_results[MRR]),
        'best_params': trial.params
    }
    with open('results_gemma.json', 'w') as f:
        json.dump(results_data, f, indent=4)
    print("Saved best trial results to results_gemma.json!", flush=True)

if __name__ == "__main__":
    main()
