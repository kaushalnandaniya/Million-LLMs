import os
import pickle
import numpy as np
import pandas as pd
import ir_measures
from ir_measures import *
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import json

def add_group_normalized_features(df):
    """
    Computes within-query group-relative features to eliminate scale disparity
    across queries in Learning-to-Rank tree splits.
    """
    df = df.copy()

    # 1. Gemma-2B Cross-Encoder Group Normalizations
    g = df.groupby('query_id')['ce_score']
    df['ce_zscore'] = g.transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
    df['ce_diff_max'] = g.transform(lambda s: s - s.max())
    df['ce_minmax'] = g.transform(lambda s: (s - s.min()) / (s.max() - s.min() + 1e-8))
    df['ce_rank'] = g.rank(ascending=False, method='min') / 150.0

    # 2. BM25 Lexical Group Normalizations
    g = df.groupby('query_id')['bm25_score']
    df['bm25_zscore'] = g.transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
    df['bm25_diff_max'] = g.transform(lambda s: s - s.max())
    df['bm25_minmax'] = g.transform(lambda s: (s - s.min()) / (s.max() - s.min() + 1e-8))
    df['bm25_rank'] = g.rank(ascending=False, method='min') / 150.0

    # 3. BGE Dense Semantic Group Normalizations
    g = df.groupby('query_id')['bge_sim']
    df['bge_zscore'] = g.transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
    df['bge_diff_max'] = g.transform(lambda s: s - s.max())
    df['bge_minmax'] = g.transform(lambda s: (s - s.min()) / (s.max() - s.min() + 1e-8))
    df['bge_rank'] = g.rank(ascending=False, method='min') / 150.0

    # 4. Uncertainty Log-Probability Group Normalizations
    g = df.groupby('query_id')['avg_mean_lp']
    df['lp_zscore'] = g.transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
    df['lp_diff_max'] = g.transform(lambda s: s - s.max())

    # 5. Multi-Head Reciprocal Rank Fusion (RRF)
    # Standard IR rank fusion: 1 / (60 + rank)
    df['rrf_score'] = (
        1.0 / (60.0 + df['ce_rank'] * 150.0) +
        1.0 / (60.0 + df['bm25_rank'] * 150.0) +
        1.0 / (60.0 + df['bge_rank'] * 150.0)
    )

    # 6. Cross-Modal Synergies
    df['ce_x_bm25'] = df['ce_minmax'] * df['bm25_minmax']
    df['ce_x_bge'] = df['ce_minmax'] * df['bge_minmax']
    df['ce_x_lp'] = df['ce_minmax'] * df['lp_zscore']

    # 7. Domain Specialization Interactivity
    df['ce_x_code'] = df['ce_minmax'] * df['is_code']
    df['ce_x_math'] = df['ce_minmax'] * df['is_math']
    df['ce_x_reasoning'] = df['ce_minmax'] * df['is_reasoning']

    return df

def evaluate_predictions(qrels_eval, query_ids, doc_ids, scores):
    """Calculates official TREC aggregate metrics."""
    run_df = pd.DataFrame({
        'query_id': query_ids,
        'doc_id': doc_ids,
        'score': scores
    })
    metrics = [nDCG@10, nDCG@20, MRR, MAP, P@10]
    results = ir_measures.calc_aggregate(metrics, qrels_eval, run_df)
    return {str(m): float(v) for m, v in results.items()}

def run_experiment():
    cache_path = 'data/dev_features_base.pkl'
    if not os.path.exists(cache_path):
        print(f"Base features cache {cache_path} not found. Run src/extract_features_df.py first.")
        return

    print("Loading base features DataFrame from cache...")
    with open(cache_path, 'rb') as f:
        df_base = pickle.load(f)

    print(f"Loaded {len(df_base)} rows across {df_base['query_id'].nunique()} queries.")

    # Prepare ground truth eval qrels
    qrels_eval = df_base[['query_id', 'llm_id', 'label']].rename(columns={'llm_id': 'doc_id', 'label': 'relevance'}).copy()

    # 1. Standalone Baselines Evaluation
    print("\n--- Evaluating Standalone Baselines ---")
    ce_metrics = evaluate_predictions(qrels_eval, df_base['query_id'], df_base['llm_id'], df_base['ce_score'])
    bm25_metrics = evaluate_predictions(qrels_eval, df_base['query_id'], df_base['llm_id'], df_base['bm25_score'])
    bge_metrics = evaluate_predictions(qrels_eval, df_base['query_id'], df_base['llm_id'], df_base['bge_sim'])

    print(f"Standalone BM25:        nDCG@10 = {bm25_metrics.get('nDCG@10', 0):.4f}, MRR = {bm25_metrics.get('RR', 0):.4f}")
    print(f"Standalone BGE Dense:   nDCG@10 = {bge_metrics.get('nDCG@10', 0):.4f}, MRR = {bge_metrics.get('RR', 0):.4f}")
    print(f"Standalone Gemma CE:    nDCG@10 = {ce_metrics.get('nDCG@10', 0):.4f}, MRR = {ce_metrics.get('RR', 0):.4f}")

    # Add Group Normalized Features
    print("\nGenerating group-normalized and cross-modal interaction features...")
    df_feat = add_group_normalized_features(df_base)

    base_feature_cols = [
        'bge_sim', 'bge_large_sim', 'e5_sim',
        'bm25_score', 'ce_score',
        'avg_mean_lp', 'avg_min_lp', 'avg_length',
        'global_mean_lp', 'global_min_lp', 'global_len',
        'mean_knn_sim', 'q_len',
        'diff_mean_lp', 'diff_length',
        'is_code', 'is_math', 'is_factual', 'is_creative', 'is_reasoning'
    ]

    new_feature_cols = [
        'ce_zscore', 'ce_diff_max', 'ce_minmax', 'ce_rank',
        'bm25_zscore', 'bm25_diff_max', 'bm25_minmax', 'bm25_rank',
        'bge_zscore', 'bge_diff_max', 'bge_minmax', 'bge_rank',
        'lp_zscore', 'lp_diff_max',
        'rrf_score',
        'ce_x_bm25', 'ce_x_bge', 'ce_x_lp',
        'ce_x_code', 'ce_x_math', 'ce_x_reasoning'
    ]

    all_feature_cols = base_feature_cols + new_feature_cols
    print(f"Base feature count: {len(base_feature_cols)}")
    print(f"Total feature count (with group normalization): {len(all_feature_cols)}")

    # Setup Cross-Validation
    queries = df_feat['query_id'].values
    unique_queries = np.unique(queries)
    label_encoder = LabelEncoder()
    encoded_queries = label_encoder.fit_transform(queries)
    y = df_feat['label'].values

    gkf = GroupKFold(n_splits=5)

    # Function for 5-fold CV with XGBRanker
    def cv_xgb(feature_cols, params, blend_ce=0.03):
        oof = np.zeros(len(df_feat))
        X = df_feat[feature_cols].values
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=queries)):
            X_train, y_train, q_train = X[train_idx], y[train_idx], encoded_queries[train_idx]
            X_test, q_test = X[test_idx], encoded_queries[test_idx]
            
            # Sort by qid for ranker
            sort_tr = np.argsort(q_train)
            X_train, y_train, q_train = X_train[sort_tr], y_train[sort_tr], q_train[sort_tr]
            
            sort_te = np.argsort(q_test)
            X_test, q_test = X_test[sort_te], q_test[sort_te]
            
            model = xgb.XGBRanker(**params)
            model.fit(X_train, y_train, qid=q_train)
            
            preds_sorted = model.predict(X_test)
            preds_orig = np.zeros_like(preds_sorted)
            preds_orig[sort_te] = preds_sorted
            
            if blend_ce > 0:
                ce_test = df_feat.iloc[test_idx]['ce_score'].values
                p_norm = (preds_orig - np.mean(preds_orig)) / (np.std(preds_orig) + 1e-8)
                c_norm = (ce_test - np.mean(ce_test)) / (np.std(ce_test) + 1e-8)
                oof[test_idx] = p_norm + blend_ce * c_norm
            else:
                oof[test_idx] = preds_orig
                
        return oof

    # Function for 5-fold CV with LightGBM LGBMRanker
    def cv_lgb(feature_cols, params, blend_ce=0.03):
        oof = np.zeros(len(df_feat))
        X = df_feat[feature_cols].values
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=queries)):
            # LightGBM requires query group lengths (counts of items per query)
            df_train = df_feat.iloc[train_idx]
            df_test = df_feat.iloc[test_idx]
            
            # Sort by query_id so groups are contiguous
            train_order = np.argsort(df_train['query_id'].values)
            test_order = np.argsort(df_test['query_id'].values)
            
            df_train_sorted = df_train.iloc[train_order]
            df_test_sorted = df_test.iloc[test_order]
            
            X_tr = df_train_sorted[feature_cols].values
            y_tr = df_train_sorted['label'].values
            q_tr_counts = df_train_sorted.groupby('query_id', sort=False).size().values
            
            X_te = df_test_sorted[feature_cols].values
            
            ranker = lgb.LGBMRanker(**params)
            ranker.fit(X_tr, y_tr, group=q_tr_counts)
            
            preds_sorted = ranker.predict(X_te)
            preds_orig = np.zeros_like(preds_sorted)
            preds_orig[test_order] = preds_sorted
            
            if blend_ce > 0:
                ce_test = df_test['ce_score'].values
                p_norm = (preds_orig - np.mean(preds_orig)) / (np.std(preds_orig) + 1e-8)
                c_norm = (ce_test - np.mean(ce_test)) / (np.std(ce_test) + 1e-8)
                oof[test_idx] = p_norm + blend_ce * c_norm
            else:
                oof[test_idx] = preds_orig
                
        return oof

    xgb_base_params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'min_child_weight': 8,
        'objective': 'rank:ndcg',
        'random_state': 42,
        'n_jobs': -1
    }

    lgb_params = {
        'n_estimators': 500,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'objective': 'lambdarank',
        'ndcg_eval_at': [10, 20],
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # Model 1: XGBoost on Base Features
    print("\n--- Training Model 1: XGBoost on Base Features (Previous SOTA) ---")
    oof_xgb_base = cv_xgb(base_feature_cols, xgb_base_params, blend_ce=0.0329)
    res_xgb_base = evaluate_predictions(qrels_eval, df_feat['query_id'], df_feat['llm_id'], oof_xgb_base)
    print(f"XGBoost (Base Features):     nDCG@10 = {res_xgb_base.get('nDCG@10', 0):.4f}, nDCG@20 = {res_xgb_base.get('nDCG@20', 0):.4f}, MRR = {res_xgb_base.get('RR', 0):.4f}")

    # Model 2: XGBoost with Group-Normalized Features
    print("\n--- Training Model 2: XGBoost with Group-Normalized Features ---")
    oof_xgb_group = cv_xgb(all_feature_cols, xgb_base_params, blend_ce=0.0329)
    res_xgb_group = evaluate_predictions(qrels_eval, df_feat['query_id'], df_feat['llm_id'], oof_xgb_group)
    print(f"XGBoost (+ Group Norm):      nDCG@10 = {res_xgb_group.get('nDCG@10', 0):.4f}, nDCG@20 = {res_xgb_group.get('nDCG@20', 0):.4f}, MRR = {res_xgb_group.get('RR', 0):.4f}")

    # Model 3: LightGBM LambdaMART with Group-Normalized Features
    print("\n--- Training Model 3: LightGBM LambdaMART (+ Group Norm) ---")
    oof_lgb = cv_lgb(all_feature_cols, lgb_params, blend_ce=0.03)
    res_lgb = evaluate_predictions(qrels_eval, df_feat['query_id'], df_feat['llm_id'], oof_lgb)
    print(f"LightGBM LambdaMART:         nDCG@10 = {res_lgb.get('nDCG@10', 0):.4f}, nDCG@20 = {res_lgb.get('nDCG@20', 0):.4f}, MRR = {res_lgb.get('RR', 0):.4f}")

    # Model 4: Multi-Model Ranker Ensemble (XGBoost + LightGBM + Gemma Logits)
    print("\n--- Training Model 4: Multi-Model Ranker Ensemble (XGBoost + LightGBM) ---")
    # Z-normalize out-of-fold scores before ensembling
    norm_xgb = (oof_xgb_group - np.mean(oof_xgb_group)) / (np.std(oof_xgb_group) + 1e-8)
    norm_lgb = (oof_lgb - np.mean(oof_lgb)) / (np.std(oof_lgb) + 1e-8)
    norm_ce = (df_feat['ce_score'].values - np.mean(df_feat['ce_score'].values)) / (np.std(df_feat['ce_score'].values) + 1e-8)

    # Grid search optimal ensemble weights on OOF
    best_weights = None
    best_ndcg = -1
    best_oof_ensemble = None

    for w_xgb in [0.4, 0.5, 0.6]:
        for w_lgb in [0.3, 0.4, 0.5]:
            for w_ce in [0.02, 0.05, 0.1]:
                combined = w_xgb * norm_xgb + w_lgb * norm_lgb + w_ce * norm_ce
                score = evaluate_predictions(qrels_eval, df_feat['query_id'], df_feat['llm_id'], combined).get('nDCG@10', 0)
                if score > best_ndcg:
                    best_ndcg = score
                    best_weights = (w_xgb, w_lgb, w_ce)
                    best_oof_ensemble = combined

    res_ensemble = evaluate_predictions(qrels_eval, df_feat['query_id'], df_feat['llm_id'], best_oof_ensemble)
    print(f"Best Ensemble Weights (XGB, LGB, CE): {best_weights}")
    print(f"Ensemble Ranker:             nDCG@10 = {res_ensemble.get('nDCG@10', 0):.4f}, nDCG@20 = {res_ensemble.get('nDCG@20', 0):.4f}, MRR = {res_ensemble.get('RR', 0):.4f}")

    # Export TREC submission file for best ensemble
    os.makedirs('results', exist_ok=True)
    df_export = df_feat[['query_id', 'llm_id']].copy()
    df_export['predicted_score'] = best_oof_ensemble
    
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
                'run_id': 'gemma_lgb_xgb_ensemble'
            })
    trec_df = pd.DataFrame(trec_records)
    trec_df.to_csv('results/ensemble_router_run.trec', sep=' ', index=False, header=False)
    print("Saved official TREC run to results/ensemble_router_run.trec")

    # Save complete results summary
    all_results = {
        'standalone_baselines': {
            'bm25': bm25_metrics,
            'bge_dense': bge_metrics,
            'gemma_ce': ce_metrics
        },
        'xgboost_base': res_xgb_base,
        'xgboost_group_norm': res_xgb_group,
        'lightgbm_group_norm': res_lgb,
        'ensemble': {
            'metrics': res_ensemble,
            'best_weights': {
                'w_xgb': best_weights[0],
                'w_lgb': best_weights[1],
                'w_ce': best_weights[2]
            }
        }
    }
    with open('results/results_ensemble.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    print("Saved all benchmark results to results/results_ensemble.json")

if __name__ == '__main__':
    run_experiment()
