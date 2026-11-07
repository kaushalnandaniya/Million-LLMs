import os
import pickle
import json
import optuna
import numpy as np
import pandas as pd
import ir_measures
from ir_measures import *
from sklearn.model_selection import GroupKFold
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import xgboost as xgb
import lightgbm as lgb

def prepare_enhanced_features():
    enhanced_path = 'data/dev_features_enhanced.pkl'
    if os.path.exists(enhanced_path):
        print(f"Loading cached enhanced features from {enhanced_path}...")
        with open(enhanced_path, 'rb') as f:
            return pickle.load(f)

    print("Building enhanced features with Category-Aware Topic Semantics...")
    with open('data/dev_features_base.pkl', 'rb') as f:
        df = pickle.load(f)

    dev_queries = []
    with open('data/data/llm_dev_data.tsv') as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 2:
                dev_queries.append({'query_id': p[0], 'text': p[1]})
    dev_queries_df = pd.DataFrame(dev_queries)

    unique_qids = dev_queries_df['query_id'].tolist()
    unique_qtexts = dev_queries_df['text'].tolist()
    llm_ids = df['llm_id'].unique().tolist()
    clean_llm_names = [lid.replace('_', ' ') for lid in llm_ids]

    print("Computing Category-Aware Topic Semantics via BGE-Large and E5-Large...")
    m_bge = SentenceTransformer('BAAI/bge-large-en-v1.5')
    m_e5 = SentenceTransformer('intfloat/e5-large-v2')

    q_bge = m_bge.encode(unique_qtexts, convert_to_tensor=True, show_progress_bar=False)
    id_bge = m_bge.encode(clean_llm_names, convert_to_tensor=True, show_progress_bar=False)
    sim_bge_name = util.cos_sim(q_bge, id_bge).cpu().numpy()

    q_e5 = m_e5.encode(['query: ' + q for q in unique_qtexts], convert_to_tensor=True, show_progress_bar=False)
    id_e5 = m_e5.encode(['passage: ' + n for n in clean_llm_names], convert_to_tensor=True, show_progress_bar=False)
    sim_e5_name = util.cos_sim(q_e5, id_e5).cpu().numpy()

    # BM25 on clean category topic names
    tokenized_names = [n.lower().split() for n in clean_llm_names]
    bm25_names = BM25Okapi(tokenized_names)
    bm25_name_scores = np.zeros((len(unique_qtexts), len(clean_llm_names)))
    for i, q in enumerate(unique_qtexts):
        bm25_name_scores[i] = bm25_names.get_scores(q.lower().split())

    qid_to_idx = {qid: i for i, qid in enumerate(unique_qids)}
    lid_to_idx = {lid: i for i, lid in enumerate(llm_ids)}

    df['name_sim_bge'] = [sim_bge_name[qid_to_idx[row['query_id']], lid_to_idx[row['llm_id']]] for _, row in df.iterrows()]
    df['name_sim_e5'] = [sim_e5_name[qid_to_idx[row['query_id']], lid_to_idx[row['llm_id']]] for _, row in df.iterrows()]
    df['name_bm25'] = [bm25_name_scores[qid_to_idx[row['query_id']], lid_to_idx[row['llm_id']]] for _, row in df.iterrows()]

    with open(enhanced_path, 'wb') as f:
        pickle.dump(df, f)
    print(f"Successfully cached enhanced features to {enhanced_path}!")
    return df

def main():
    df = prepare_enhanced_features()
    qrels_eval = df[['query_id', 'llm_id', 'label']].rename(columns={'llm_id': 'doc_id', 'label': 'relevance'}).copy()

    queries = df['query_id'].values
    q_codes = df['query_id'].astype('category').cat.codes.values
    y = df['label'].values
    gkf = GroupKFold(n_splits=5)

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

    print(f"Running Optuna Bayesian Optimization over {len(feature_cols)} features...")

    def evaluate_xgb(param, ce_blend=0.0, name_blend=0.0):
        oof = np.zeros(len(df))
        X = df[feature_cols].values
        
        for train_idx, test_idx in gkf.split(df, y, groups=queries):
            X_tr, y_tr, q_tr = X[train_idx], y[train_idx], q_codes[train_idx]
            X_te, q_te = X[test_idx], q_codes[test_idx]
            
            s_tr = np.argsort(q_tr)
            s_te = np.argsort(q_te)
            
            m = xgb.XGBRanker(**param)
            m.fit(X_tr[s_tr], y_tr[s_tr], qid=q_tr[s_tr])
            
            preds = np.zeros(len(test_idx))
            preds[s_te] = m.predict(X_te[s_te])
            
            p_norm = (preds - np.mean(preds)) / (np.std(preds) + 1e-8)
            combined = p_norm
            
            if ce_blend > 0:
                ce_test = df.iloc[test_idx]['ce_score'].values
                c_norm = (ce_test - np.mean(ce_test)) / (np.std(ce_test) + 1e-8)
                combined = combined + ce_blend * c_norm
                
            if name_blend > 0:
                name_test = df.iloc[test_idx]['name_sim_bge'].values
                n_norm = (name_test - np.mean(name_test)) / (np.std(name_test) + 1e-8)
                combined = combined + name_blend * n_norm
                
            oof[test_idx] = combined
            
        return oof

    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 700),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 4, 16),
            'objective': 'rank:ndcg',
            'random_state': 42,
            'n_jobs': -1
        }
        ce_blend = trial.suggest_float('ce_blend', 0.0, 0.1)
        name_blend = trial.suggest_float('name_blend', 0.0, 0.2)

        oof = evaluate_xgb(param, ce_blend, name_blend)
        run_df = pd.DataFrame({'query_id': queries, 'doc_id': df['llm_id'].values, 'score': oof})
        res = ir_measures.calc_aggregate([nDCG@20], qrels_eval, run_df)
        return res[nDCG@20]

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=40)

    print("\n" + "="*60)
    print("BEST ENHANCED TRIAL RESULTS:")
    best = study.best_trial
    print(f"  Best nDCG@20: {best.value:.4f}")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print("="*60 + "\n")

    # Evaluate final best model on all metrics
    best_ce_blend = best.params.get('ce_blend', 0.0)
    best_name_blend = best.params.get('name_blend', 0.0)
    best_xgb_params = {k: v for k, v in best.params.items() if k not in ['ce_blend', 'name_blend']}
    best_xgb_params['objective'] = 'rank:ndcg'
    best_xgb_params['random_state'] = 42
    best_xgb_params['n_jobs'] = -1

    final_oof = evaluate_xgb(best_xgb_params, best_ce_blend, best_name_blend)
    final_run_df = pd.DataFrame({'query_id': queries, 'doc_id': df['llm_id'].values, 'score': final_oof})

    full_metrics = [nDCG@10, nDCG@20, MRR, MAP, P@10]
    agg_res = ir_measures.calc_aggregate(full_metrics, qrels_eval, final_run_df)

    print("--- Final Enhanced Multi-Modal Pipeline Evaluation ---")
    for m, val in agg_res.items():
        print(f"  {m}: {val:.4f}")

    # Export official TREC run file
    os.makedirs('results', exist_ok=True)
    df_export = df[['query_id', 'llm_id']].copy()
    df_export['predicted_score'] = final_oof

    trec_records = []
    for qid, grp in df_export.groupby('query_id'):
        grp_sorted = grp.sort_values(by='predicted_score', ascending=False)
        for rank, (_, row) in enumerate(grp_sorted.iterrows()):
            trec_records.append({
                'query_id': qid,
                'Q0': 'Q0',
                'llm_id': row['llm_id'],
                'rank': rank + 1,
                'score': row['predicted_score'],
                'run_id': 'topic_aware_gemma_router'
            })
    trec_df = pd.DataFrame(trec_records)
    trec_path = 'results/enhanced_router_run.trec'
    trec_df.to_csv(trec_path, sep=' ', index=False, header=False)
    print(f"Saved official TREC run to {trec_path}")

    # Save summary json
    results_data = {
        'best_ndcg_10': float(agg_res[nDCG@10]),
        'best_ndcg_20': float(agg_res[nDCG@20]),
        'best_mrr': float(agg_res[MRR]),
        'best_map': float(agg_res[MAP]),
        'best_p10': float(agg_res[P@10]),
        'best_params': best.params
    }
    with open('results/results_enhanced.json', 'w') as f:
        json.dump(results_data, f, indent=4)
    print("Saved results to results/results_enhanced.json!")

if __name__ == '__main__':
    main()
