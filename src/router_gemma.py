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
    llm_responses_raw = {}       # llm_id -> [individual response strings]
    
    for qid, q_data in data.items():
        q_text = q_data.get('query', '')
        discovery_queries[qid] = q_text
        discovery_stats[qid] = {}
        
        responses = q_data.get('responses', {})
        for llm_id, res_obj in responses.items():
            if llm_id not in llm_global_stats_raw:
                llm_global_stats_raw[llm_id] = {'mean_lps': [], 'min_lps': [], 'lengths': []}
                llm_profiles[llm_id] = []
                llm_responses_raw[llm_id] = []
                
            resp_text = res_obj.get('response', '')
            logprobs = res_obj.get('logprobs', [])
            
            # Increase profile size limit to 50 for a richer semantic representation
            if resp_text and resp_text.lower() != "i don't know":
                if len(llm_profiles[llm_id]) < 50:
                    llm_profiles[llm_id].append(resp_text)
                    llm_responses_raw[llm_id].append(resp_text)
                    
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
        
    return discovery_queries, discovery_stats, llm_global_stats, llm_profiles, llm_responses_raw

def select_best_response_chunk(query_tokens, responses, max_chars=1200):
    """
    Selects the single most semantically relevant historical response from the LLM
    instead of truncating an arbitrary concatenation of up to 50 responses.
    """
    if not responses:
        return ""
    if len(responses) == 1:
        return responses[0][:max_chars]
        
    q_set = set(w.lower() for w in query_tokens if len(w) > 2)
    best_idx = 0
    best_overlap = -1
    for i, resp in enumerate(responses):
        resp_lower = resp.lower()
        overlap = sum(1 for w in q_set if w in resp_lower)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i
            
    return responses[best_idx][:max_chars]

def extract_query_domain_features(qtext):
    """
    Extracts 5 domain intent indicators (Coding, Math, Factual QA, Creative, Reasoning)
    to enable the Learning-to-Rank engine to route queries based on domain specialization.
    """
    import re
    q_lower = qtext.lower()
    q_words = set(re.findall(r'[a-z0-9_+#\.]+', q_lower))
    
    # 1. Code / Programming
    code_keywords = {
        'code', 'python', 'java', 'c++', 'javascript', 'html', 'css', 'sql', 'function',
        'def', 'class', 'bug', 'error', 'compile', 'debugging', 'algorithm', 'return',
        'api', 'import', 'loop', 'syntax', 'array', 'string', 'int', 'null', 'script',
        'regex', 'git', 'variable', 'method', 'pointer', 'docker', 'bash', 'terminal',
        'quicksort', 'recursion', 'sorting', 'programming'
    }
    is_code = 1.0 if any(w in code_keywords for w in q_words) or any(k in q_lower for k in ['def ', 'import ', 'print(', 'console.log']) else 0.0
    
    # 2. Math / Quantitative
    math_keywords = {
        'math', 'calculate', 'solve', 'equation', 'probability', 'integral', 'matrix',
        'theorem', 'sum', 'formula', 'geometry', 'algebra', 'prime', 'derivative',
        'multiply', 'divide', 'ratio', 'percentage', 'fraction', 'logarithm', 'polynomial'
    }
    is_math = 1.0 if any(w in math_keywords for w in q_words) or any(c in q_lower for c in ['+', '=', '*', '^', 'sqrt']) else 0.0
    
    # 3. Factual QA / Lookup
    factual_starters = ('who is', 'who was', 'where is', 'where was', 'when was', 'when did', 'what year', 'capital of', 'which year')
    is_factual = 1.0 if any(q_lower.startswith(s) for s in factual_starters) or 'birthplace' in q_lower or 'history of' in q_lower else 0.0
    
    # 4. Creative Writing
    creative_cues = {'write', 'story', 'poem', 'essay', 'draft', 'rewrite', 'compose', 'dialogue', 'lyrics', 'creative'}
    is_creative = 1.0 if any(w in creative_cues for w in q_words) else 0.0
    
    # 5. Explanatory / Causal Reasoning
    reasoning_cues = {'why', 'explain', 'how does', 'cause', 'reason', 'difference between', 'compare', 'analyze', 'mechanism'}
    is_reasoning = 1.0 if any(w in reasoning_cues for w in q_words) or q_lower.startswith('why') or q_lower.startswith('how') else 0.0
    
    return [is_code, is_math, is_factual, is_creative, is_reasoning]

def build_features(dev_queries_df, qrels_df, discovery_queries, discovery_stats, llm_global_stats, models, llm_profiles, llm_responses_raw=None, k_nearest=5):
    import torch
    from sentence_transformers import util
    from sentence_transformers.cross_encoder import CrossEncoder
    from rank_bm25 import BM25Okapi

    print("Precomputing profile embeddings and Lexical BM25 indices...")
    llm_ids = list(llm_profiles.keys())
    profile_texts = list(llm_profiles.values())
    
    profile_embeddings_dict = {}
    for m_name, m in models.items():
        profile_embeddings_dict[m_name] = m.encode(profile_texts, show_progress_bar=False, convert_to_tensor=True)
        
    tokenized_corpus = [doc.lower().split(" ") for doc in profile_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    import math
    import concurrent.futures
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_file = "data/gemma_ce_scores.pt"
    checkpoint_file = "data/gemma_ce_scores_checkpoint.pt"
    
    if os.path.exists(cache_file):
        print(f"Found cached Gemma-2B scores at {cache_file}! Loading instantly...")
        qtext_llm_to_ce = torch.load(cache_file)
    else:
        print("Precomputing Gemma-2B Reranker scores for all valid query-LLM pairs using semantic chunk selection...")
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
                # Granular Response-Level Chunk Selection (fixes profile dilution)
                if llm_responses_raw and llm_id in llm_responses_raw:
                    q_toks = qtext.lower().split()
                    best_chunk = select_best_response_chunk(q_toks, llm_responses_raw[llm_id], max_chars=1200)
                    ce_inputs.append((qtext, best_chunk))
                else:
                    ce_inputs.append((qtext, llm_profiles[llm_id][:1500])) # Truncate fallback
                
        print(f"Scoring {len(ce_inputs)} pairs using Gemma-2B Reranker (balanced across BOTH GPUs)...")
        
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-gemma')
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # device_map='balanced' spreads the 2B model across both GPUs
        model_ce = AutoModelForCausalLM.from_pretrained('BAAI/bge-reranker-v2-gemma', device_map='balanced', torch_dtype=torch.float16)
        model_ce.eval()
        
        yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
        
        def get_inputs(pairs, max_length=1024):
            prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
            texts = []
            for query, passage in pairs:
                texts.append(f"A: {query}\nB: {passage}\n{prompt}")
            return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

        batch_size = 16  # 16 fits within the remaining VRAM
        ce_scores = []
        start_batch = 0
        if os.path.exists(checkpoint_file):
            try:
                ce_scores = torch.load(checkpoint_file)
                start_batch = len(ce_scores) // batch_size
                print(f"Resuming Gemma scoring from checkpoint: {len(ce_scores)} pairs ({start_batch} batches) already done!", flush=True)
            except Exception as e:
                print(f"Could not load checkpoint ({e}), starting from scratch.", flush=True)
                ce_scores = []
                start_batch = 0
        
        from tqdm import tqdm
        num_batches = math.ceil(len(ce_inputs) / batch_size)
        with torch.no_grad():
            for i in tqdm(range(start_batch, num_batches), desc="Gemma Reranking Batches", initial=start_batch, total=num_batches):
                batch_pairs = ce_inputs[i*batch_size : (i+1)*batch_size]
                inputs = get_inputs(batch_pairs)
                inputs = {k: v.to(model_ce.device) for k, v in inputs.items()}
                
                # Extract logit for "Yes" token from the last position
                scores = model_ce(**inputs, return_dict=True).logits[:, -1, yes_loc].view(-1, ).float()
                ce_scores.extend(scores.cpu().tolist())

                # Checkpoint every 50 batches (~1.5 minutes)
                if (i + 1) % 50 == 0 or (i + 1) == num_batches:
                    torch.save(ce_scores, checkpoint_file)

        for i, key in enumerate(ce_keys):
            qtext_llm_to_ce[key] = float(ce_scores[i])
            
        torch.save(qtext_llm_to_ce, cache_file)
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        print(f"Saved complete Gemma scores to {cache_file}!", flush=True)
            
        # Free up VRAM so XGBoost training has no issues
        import gc
        del model_ce
        torch.cuda.empty_cache()
        gc.collect()
    
    print("Embedding discovery queries for KNN search (using primary bge model)...")
    disc_qids = list(discovery_queries.keys())
    disc_qtexts = list(discovery_queries.values())
    disc_q_embeddings = models['bge'].encode(disc_qtexts, show_progress_bar=True, convert_to_tensor=True)
    
    features = []
    labels = []
    groups = []
    
    print(f"Building KNN Query-Aware features for {len(qrels_df)} query-LLM pairs using GPU...")
    
    unique_qtexts = dev_queries_df['text'].unique().tolist()
    q_embeddings_dict = {}
    for m_name, m in models.items():
        q_embeddings_dict[m_name] = m.encode(unique_qtexts, show_progress_bar=False, convert_to_tensor=True)
        
    print("Precomputing similarities on GPU...")
    # Compute similarity between all unique queries and all discovery queries at once
    sim_matrix = util.cos_sim(q_embeddings_dict['bge'], disc_q_embeddings)
    top_k_values, top_k_indices = torch.topk(sim_matrix, k=k_nearest, dim=1)
    qtext_to_top_k_idx = {qtext: top_k_indices[i].cpu().numpy() for i, qtext in enumerate(unique_qtexts)}
    qtext_to_top_k_vals = {qtext: top_k_values[i].cpu().numpy() for i, qtext in enumerate(unique_qtexts)}
    
    # Compute similarity between all unique queries and all LLM profiles at once
    qtext_to_profile_sims = {}
    for m_name in models.keys():
        profile_sim_matrix = util.cos_sim(q_embeddings_dict[m_name], profile_embeddings_dict[m_name])
        qtext_to_profile_sims[m_name] = {qtext: profile_sim_matrix[i].cpu().numpy() for i, qtext in enumerate(unique_qtexts)}
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
            sim_scores = [float(qtext_to_profile_sims[m_name][qtext][idx]) for m_name in models.keys()]
            
            tokenized_query = qtext.lower().split(" ")
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_score = float(bm25_scores[idx])
        else:
            sim_scores = [0.0 for _ in models.keys()]
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
        domain_feats = extract_query_domain_features(qtext)
        
        feature_vec = sim_scores + [
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
        ] + domain_feats
        
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
    
    discovery_queries, discovery_stats, llm_global_stats, llm_profiles, llm_responses_raw = extract_discovery_data(f'{data_dir}/llm_discovery_metadata.json')
    
    print("Initializing Multi-Embedding Ensembles...")
    models = {
        'bge': SentenceTransformer('BAAI/bge-base-en-v1.5'),
        'bge_large': SentenceTransformer('BAAI/bge-large-en-v1.5'),
        'e5': SentenceTransformer('intfloat/e5-large-v2')
    }
    
    X, y, groups = build_features(dev_queries_df, qrels_df, discovery_queries, discovery_stats, llm_global_stats, models, llm_profiles, llm_responses_raw=llm_responses_raw, k_nearest=5)
    
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
            objective='rank:ndcg', 
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
