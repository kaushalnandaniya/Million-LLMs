import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import json

def extract_discovery_data(metadata_file):
    print("Extracting query-level LLM statistics from discovery metadata...")
    with open(metadata_file, 'r') as f:
        data = json.load(f)
        
    discovery_queries = {}       # qid -> query_text
    discovery_stats = {}         # qid -> {llm_id -> {'mean_lp': x, 'min_lp': y, 'len': z}}
    llm_global_stats_raw = {}    # llm_id -> {'mean_lps': [], 'min_lps': [], 'lengths': []}
    llm_profiles = {}            # llm_id -> [list of responses]
    
    for qid, q_data in data.items():
        q_text = q_data.get('query', '')
        discovery_queries[qid] = q_text
        discovery_stats[qid] = {}
        
        responses = q_data.get('responses', {})
        for llm_id, res_obj in responses.items():
            if llm_id not in llm_global_stats_raw:
                llm_global_stats_raw[llm_id] = {'mean_lps': [], 'min_lps': [], 'lengths': []}
                llm_profiles[llm_id] = []
                
            resp_text = res_obj.get('response', '')
            logprobs = res_obj.get('logprobs', [])
            
            # Increase profile size limit to 50 for a richer semantic representation
            if resp_text and resp_text.lower() != "i don't know":
                if len(llm_profiles[llm_id]) < 50:
                    llm_profiles[llm_id].append(resp_text)
                    
            if logprobs:
                valid_lps = [float(lp) for lp in logprobs if lp is not None]
                if valid_lps:
                    mean_lp = np.mean(valid_lps)
                    min_lp = np.min(valid_lps)
                    length = len(valid_lps)
                    
                    discovery_stats[qid][llm_id] = {
                        'mean_lp': mean_lp,
                        'min_lp': min_lp,
                        'len': length
                    }
                    
                    llm_global_stats_raw[llm_id]['mean_lps'].append(mean_lp)
                    llm_global_stats_raw[llm_id]['min_lps'].append(min_lp)
                    llm_global_stats_raw[llm_id]['lengths'].append(length)
                    
    # Compute global averages as fallback
    llm_global_stats = {}
    for llm_id, raw in llm_global_stats_raw.items():
        llm_global_stats[llm_id] = {
            'mean_lp': np.mean(raw['mean_lps']) if raw['mean_lps'] else -1.0,
            'min_lp': np.mean(raw['min_lps']) if raw['min_lps'] else -5.0,
            'len': np.mean(raw['lengths']) if raw['lengths'] else 10
        }
        
    for llm_id in llm_profiles:
        llm_profiles[llm_id] = " ".join(llm_profiles[llm_id])
        
    return discovery_queries, discovery_stats, llm_global_stats, llm_profiles

def build_features(dev_queries_df, qrels_df, discovery_queries, discovery_stats, llm_global_stats, model, llm_profiles, k_nearest=5):
    import torch
    from sentence_transformers import util
    from sentence_transformers.cross_encoder import CrossEncoder
    from rank_bm25 import BM25Okapi

    print("Precomputing profile embeddings and Lexical BM25 indices...")
    llm_ids = list(llm_profiles.keys())
    profile_texts = list(llm_profiles.values())
    profile_embeddings = model.encode(profile_texts, show_progress_bar=False, convert_to_tensor=True)
    
    tokenized_corpus = [doc.lower().split(" ") for doc in profile_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    import math
    import concurrent.futures
    from sentence_transformers.cross_encoder import CrossEncoder

    print("Precomputing Cross-Encoder scores for all valid query-LLM pairs...")
    qtext_llm_to_ce = {}
    ce_inputs = []
    ce_keys = []
    for _, row in qrels_df.iterrows():
        qid = row['query_id']
        qtext_series = dev_queries_df[dev_queries_df['query_id'] == qid]['text']
        if qtext_series.empty:
            continue
        qtext = qtext_series.values[0]
        llm_id = row['llm_id']
        if llm_id in llm_profiles:
            ce_keys.append((qtext, llm_id))
            ce_inputs.append((qtext, llm_profiles[llm_id][:1500])) # Truncate to fit sequence length
            
    print(f"Scoring {len(ce_inputs)} pairs using BOTH GPUs (cuda:0 and cuda:1)...")
    ce_0 = CrossEncoder('BAAI/bge-reranker-large', device='cuda:0')
    ce_1 = CrossEncoder('BAAI/bge-reranker-large', device='cuda:1')
    
    mid = math.ceil(len(ce_inputs) / 2)
    inputs_0 = ce_inputs[:mid]
    inputs_1 = ce_inputs[mid:]
    
    def predict_chunk(ce, chunk):
        return ce.predict(chunk, batch_size=16, show_progress_bar=True)
        
    ce_scores = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f0 = executor.submit(predict_chunk, ce_0, inputs_0)
        f1 = executor.submit(predict_chunk, ce_1, inputs_1)
        ce_scores = list(f0.result()) + list(f1.result())

    for i, key in enumerate(ce_keys):
        qtext_llm_to_ce[key] = float(ce_scores[i])
    
    print("Embedding discovery queries for KNN search...")
    disc_qids = list(discovery_queries.keys())
    disc_qtexts = list(discovery_queries.values())
    disc_q_embeddings = model.encode(disc_qtexts, show_progress_bar=True, convert_to_tensor=True)
    
    features = []
    labels = []
    groups = []
    
    print(f"Building KNN Query-Aware features for {len(qrels_df)} query-LLM pairs using GPU...")
    
    unique_qtexts = dev_queries_df['text'].unique().tolist()
    q_embeddings = model.encode(unique_qtexts, show_progress_bar=False, convert_to_tensor=True)
    
    print("Precomputing similarities on GPU...")
    # Compute similarity between all unique queries and all discovery queries at once
    sim_matrix = util.cos_sim(q_embeddings, disc_q_embeddings)
    top_k_values, top_k_indices = torch.topk(sim_matrix, k=k_nearest, dim=1)
    qtext_to_top_k_idx = {qtext: top_k_indices[i].cpu().numpy() for i, qtext in enumerate(unique_qtexts)}
    qtext_to_top_k_vals = {qtext: top_k_values[i].cpu().numpy() for i, qtext in enumerate(unique_qtexts)}
    
    # Compute similarity between all unique queries and all LLM profiles at once
    profile_sim_matrix = util.cos_sim(q_embeddings, profile_embeddings)
    qtext_to_profile_sims = {qtext: profile_sim_matrix[i].cpu().numpy() for i, qtext in enumerate(unique_qtexts)}
    llm_id_to_idx = {llm_id: i for i, llm_id in enumerate(llm_ids)}
    
    for _, row in qrels_df.iterrows():
        qid = row['query_id']
        qtext_series = dev_queries_df[dev_queries_df['query_id'] == qid]['text']
        if qtext_series.empty:
            continue
        qtext = qtext_series.values[0]
        llm_id = row['llm_id']
        label = row['expertise_score']
        
        # 1. Dense Retrieval Similarity (Query vs LLM Profile)
        if llm_id in llm_id_to_idx:
            idx = llm_id_to_idx[llm_id]
            sim_score = float(qtext_to_profile_sims[qtext][idx])
            
            tokenized_query = qtext.lower().split(" ")
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_score = float(bm25_scores[idx])
        else:
            sim_score = 0.0
            bm25_score = 0.0
            
        if (qtext, llm_id) in qtext_llm_to_ce:
            ce_score = qtext_llm_to_ce[(qtext, llm_id)]
        else:
            ce_score = -10.0
            
        # 2. Query-Aware Metadata Stats (KNN)
        top_k_idx = qtext_to_top_k_idx[qtext]
        top_k_vals = qtext_to_top_k_vals[qtext]
        mean_knn_sim = float(np.mean(top_k_vals))
        
        mean_lps, min_lps, lengths = [], [], []
        weights = []
        for i, idx in enumerate(top_k_idx):
            past_qid = disc_qids[idx]
            stats = discovery_stats.get(past_qid, {}).get(llm_id)
            if stats:
                mean_lps.append(stats['mean_lp'])
                min_lps.append(stats['min_lp'])
                lengths.append(stats['len'])
                w = max(top_k_vals[i], 0.0) + 1e-6
                weights.append(w)
                
        global_stats = llm_global_stats.get(llm_id, {'mean_lp': -1.0, 'min_lp': -5.0, 'len': 10})
        
        if weights and sum(weights) > 0:
            norm_weights = [w / sum(weights) for w in weights]
            avg_mean_lp = np.average(mean_lps, weights=norm_weights)
            avg_min_lp = np.average(min_lps, weights=norm_weights)
            avg_length = np.average(lengths, weights=norm_weights)
        else:
            avg_mean_lp = global_stats['mean_lp']
            avg_min_lp = global_stats['min_lp']
            avg_length = global_stats['len']
            
        global_mean_lp = global_stats['mean_lp']
        global_min_lp = global_stats['min_lp']
        global_len = global_stats['len']
        
        q_len = len(qtext.split())
        
        feature_vec = [
            sim_score,
            bm25_score,
            ce_score,
            avg_mean_lp,
            avg_min_lp,
            avg_length,
            global_mean_lp,
            global_min_lp,
            global_len,
            mean_knn_sim,
            q_len,
            avg_mean_lp - global_mean_lp,
            avg_length - global_len
        ]
        
        features.append(feature_vec)
        labels.append(label)
        groups.append(qid)
        
    return np.array(features), np.array(labels), np.array(groups)


def run_improved_router():
    data_dir = 'data/data'
    if not os.path.exists(f'{data_dir}/llm_discovery_metadata.json'):
        print(f"Data not found at {data_dir}. Please run download scripts.")
        return

    dev_queries = []
    with open(f'{data_dir}/llm_dev_data.tsv', 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                dev_queries.append({'query_id': parts[0], 'text': parts[1]})
    dev_queries_df = pd.DataFrame(dev_queries)
    
    qrels_df = pd.read_csv(f'{data_dir}/llm_dev_qrels.txt', sep=' ', names=['query_id', 'zero', 'llm_id', 'expertise_score'], dtype={'query_id': str, 'llm_id': str})
    
    discovery_queries, discovery_stats, llm_global_stats, llm_profiles = extract_discovery_data(f'{data_dir}/llm_discovery_metadata.json')
    
    print("Initializing Heavy Sentence Transformer model (BAAI/bge-base-en-v1.5)...")
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    
    X, y, groups = build_features(dev_queries_df, qrels_df, discovery_queries, discovery_stats, llm_global_stats, model, llm_profiles, k_nearest=5)
    
    print("--- Training XGBoost Ranker (Learning-to-Rank) with GroupKFold ---")
    gkf = GroupKFold(n_splits=5)
    oof_predictions = np.zeros(len(y))
    
    group_encoder = LabelEncoder()
    encoded_groups = group_encoder.fit_transform(groups)
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"Training Fold {fold+1}...")
        X_train, y_train, groups_train = X[train_idx], y[train_idx], encoded_groups[train_idx]
        X_test, y_test, groups_test = X[test_idx], y[test_idx], encoded_groups[test_idx]
        
        train_sort_idx = np.argsort(groups_train)
        X_train, y_train, groups_train = X_train[train_sort_idx], y_train[train_sort_idx], groups_train[train_sort_idx]
        
        test_sort_idx = np.argsort(groups_test)
        X_test, groups_test = X_test[test_sort_idx], groups_test[test_sort_idx]
        
        xgb_model = xgb.XGBRanker(
            n_estimators=784, 
            learning_rate=0.0113, 
            max_depth=6,
            subsample=0.7534,
            colsample_bytree=0.6736,
            min_child_weight=13,
            objective='rank:pairwise', 
            random_state=42
        )
        xgb_model.fit(X_train, y_train, qid=groups_train)
        
        preds_sorted = xgb_model.predict(X_test)
        preds_unsorted = np.zeros_like(preds_sorted)
        preds_unsorted[test_sort_idx] = preds_sorted
        
        oof_predictions[test_idx] = preds_unsorted
    
    qrels_df['predicted_score'] = oof_predictions
    
    results = []
    for qid, group in qrels_df.groupby('query_id'):
        group_sorted = group.sort_values(by='predicted_score', ascending=False)
        for rank, (_, row) in enumerate(group_sorted.iterrows()):
            results.append({
                'query_id': qid,
                'Q0': 'Q0',
                'llm_id': row['llm_id'],
                'rank': rank + 1,
                'score': row['predicted_score'],
                'run_id': 'xgboost_knn_router_cv'
            })
            
    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/router_run.trec', sep=' ', index=False, header=False)
    
    print("Improved Router run complete. Results saved to results/router_run.trec")

if __name__ == "__main__":
    run_improved_router()
