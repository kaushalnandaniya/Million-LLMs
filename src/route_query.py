import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

from router_gemma import extract_discovery_data, extract_query_domain_features

class FastLLMRouter:
    def __init__(self, data_dir='data/data', model_path='models/xgb_enhanced_router.pkl', cache_path='data/router_inference_cache.pkl'):
        self.data_dir = data_dir
        self.model_path = model_path
        self.cache_path = cache_path
        self._load_encoders()
        self._load_assets()
        self._load_ranker()

    def _load_encoders(self):
        print("[Router] Loading neural encoders (BGE-Base, BGE-Large, E5-Large)...", flush=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = {
            'bge': SentenceTransformer('BAAI/bge-base-en-v1.5', device=device),
            'bge_large': SentenceTransformer('BAAI/bge-large-en-v1.5', device=device),
            'e5': SentenceTransformer('intfloat/e5-large-v2', device=device)
        }

    def _load_assets(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if os.path.exists(self.cache_path):
            print(f"[Router] Loading precomputed assets from {self.cache_path}...", flush=True)
            with open(self.cache_path, 'rb') as f:
                c = pickle.load(f)
            self.disc_queries = c['disc_queries']
            self.disc_qids = c['disc_qids']
            self.disc_stats = c['disc_stats']
            self.llm_global_stats = c['llm_global_stats']
            self.llm_ids = c['llm_ids']
            self.clean_llm_names = c['clean_llm_names']
            self.bm25_profile = c['bm25_profile']
            self.bm25_name = c['bm25_name']
            self.profile_embeddings = {k: v.to(device) for k, v in c['profile_embs'].items()}
            self.name_bge_embs = c['name_bge_embs'].to(device)
            self.name_e5_embs = c['name_e5_embs'].to(device)
            self.disc_q_embeddings = c['disc_q_embeddings'].to(device)
        else:
            print("[Router] Offline cache not found, parsing metadata directly...", flush=True)
            self.disc_queries, self.disc_stats, self.llm_global_stats, self.llm_profiles, _ = extract_discovery_data(
                f"{self.data_dir}/llm_discovery_metadata.json"
            )
            self.llm_ids = list(self.llm_profiles.keys())
            self.clean_llm_names = [lid.replace('_', ' ') for lid in self.llm_ids]
            self.profile_texts = [self.llm_profiles[lid] for lid in self.llm_ids]
            
            tokenized_profiles = [doc.lower().split() for doc in self.profile_texts]
            self.bm25_profile = BM25Okapi(tokenized_profiles)

            tokenized_names = [n.lower().split() for n in self.clean_llm_names]
            self.bm25_name = BM25Okapi(tokenized_names)

            self.profile_embeddings = {}
            for m_name, m in self.models.items():
                self.profile_embeddings[m_name] = m.encode(self.profile_texts, convert_to_tensor=True, show_progress_bar=False)

            self.name_bge_embs = self.models['bge_large'].encode(self.clean_llm_names, convert_to_tensor=True, show_progress_bar=False)
            self.name_e5_embs = self.models['e5'].encode(['passage: ' + n for n in self.clean_llm_names], convert_to_tensor=True, show_progress_bar=False)

            self.disc_qids = list(self.disc_queries.keys())
            disc_qtexts = list(self.disc_queries.values())
            self.disc_q_embeddings = self.models['bge'].encode(disc_qtexts, convert_to_tensor=True, show_progress_bar=False)

        self.llm_id_to_idx = {lid: i for i, lid in enumerate(self.llm_ids)}

    def _load_ranker(self):
        self.bundle = None
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.bundle = pickle.load(f)
            print(f"[Router] Successfully loaded trained production ranker from {self.model_path}!", flush=True)
        else:
            print(f"[Router] Warning: Model bundle not found at {self.model_path}. Using fallback heuristic ranking.", flush=True)

    def route(self, query_text, top_k=5):
        # 1. Domain Intent Analysis
        domain_feats = extract_query_domain_features(query_text)
        domains = []
        if domain_feats[0]: domains.append("Coding")
        if domain_feats[1]: domains.append("Math/Quantitative")
        if domain_feats[2]: domains.append("Factual QA")
        if domain_feats[3]: domains.append("Creative Writing")
        if domain_feats[4]: domains.append("Reasoning/Explanatory")
        domain_str = ", ".join(domains) if domains else "General Knowledge"

        # 2. Category-Aware Topic Semantic Similarities
        q_bge_large = self.models['bge_large'].encode(query_text, convert_to_tensor=True)
        sim_name_bge = util.cos_sim(q_bge_large, self.name_bge_embs)[0].cpu().numpy()

        q_e5_large = self.models['e5'].encode('query: ' + query_text, convert_to_tensor=True)
        sim_name_e5 = util.cos_sim(q_e5_large, self.name_e5_embs)[0].cpu().numpy()

        q_tokens = query_text.lower().split()
        bm25_name_scores = np.array(self.bm25_name.get_scores(q_tokens))

        # 3. Profile Semantic & Lexical Similarities
        q_embs = {m_name: m.encode(query_text, convert_to_tensor=True) for m_name, m in self.models.items()}
        sims = {m_name: util.cos_sim(q_embs[m_name], self.profile_embeddings[m_name])[0].cpu().numpy() for m_name in self.models}
        bm25_profile_scores = np.array(self.bm25_profile.get_scores(q_tokens))

        # 4. KNN Discovery Uncertainty
        knn_sims = util.cos_sim(q_embs['bge'], self.disc_q_embeddings)[0]
        top_k_vals, top_k_idx = torch.topk(knn_sims, k=5)
        top_k_vals = top_k_vals.cpu().numpy()
        top_k_idx = top_k_idx.cpu().numpy()
        mean_knn_sim = float(np.mean(top_k_vals))

        rows = []
        for i, lid in enumerate(self.llm_ids):
            # Uncertainty aggregation
            mean_lps, min_lps, lengths, weights = [], [], [], []
            for j, d_idx in enumerate(top_k_idx):
                stats = self.disc_stats.get(self.disc_qids[d_idx], {}).get(lid)
                if stats:
                    mean_lps.append(stats['mean_lp'])
                    min_lps.append(stats['min_lp'])
                    lengths.append(stats['len'])
                    weights.append(max(top_k_vals[j], 0.0) + 1e-6)

            g_stats = self.llm_global_stats.get(lid, {'mean_lp': -1.0, 'min_lp': -5.0, 'len': 10})
            if weights and sum(weights) > 0:
                norm_w = [w / sum(weights) for w in weights]
                avg_mlp = np.average(mean_lps, weights=norm_w)
                avg_minlp = np.average(min_lps, weights=norm_w)
                avg_l = np.average(lengths, weights=norm_w)
            else:
                avg_mlp = g_stats['mean_lp']
                avg_minlp = g_stats['min_lp']
                avg_l = g_stats['len']

            rows.append({
                'llm_id': lid,
                'name_sim_bge': float(sim_name_bge[i]),
                'name_sim_e5': float(sim_name_e5[i]),
                'name_bm25': float(bm25_name_scores[i]),
                'bge_sim': float(sims['bge'][i]),
                'bge_large_sim': float(sims['bge_large'][i]),
                'e5_sim': float(sims['e5'][i]),
                'bm25_score': float(bm25_profile_scores[i]),
                'ce_score': float(sim_name_bge[i] * 10.0), # Proxy logit
                'avg_mean_lp': float(avg_mlp),
                'avg_min_lp': float(avg_minlp),
                'avg_length': float(avg_l),
                'global_mean_lp': float(g_stats['mean_lp']),
                'global_min_lp': float(g_stats['min_lp']),
                'global_len': float(g_stats['len']),
                'mean_knn_sim': mean_knn_sim,
                'q_len': len(q_tokens),
                'diff_mean_lp': float(avg_mlp - g_stats['mean_lp']),
                'diff_length': float(avg_l - g_stats['len']),
                'is_code': domain_feats[0],
                'is_math': domain_feats[1],
                'is_factual': domain_feats[2],
                'is_creative': domain_feats[3],
                'is_reasoning': domain_feats[4],
            })

        df_cand = pd.DataFrame(rows)

        if self.bundle:
            model = self.bundle['model']
            cols = self.bundle['feature_cols']
            ce_blend = self.bundle.get('ce_blend', 0.0)
            name_blend = self.bundle.get('name_blend', 0.0)

            X_mat = df_cand[cols].values
            raw_preds = model.predict(X_mat)

            p_norm = (raw_preds - np.mean(raw_preds)) / (np.std(raw_preds) + 1e-8)
            combined = p_norm

            if ce_blend > 0:
                ce = df_cand['ce_score'].values
                c_norm = (ce - np.mean(ce)) / (np.std(ce) + 1e-8)
                combined = combined + ce_blend * c_norm

            if name_blend > 0:
                name_bge = df_cand['name_sim_bge'].values
                n_norm = (name_bge - np.mean(name_bge)) / (np.std(name_bge) + 1e-8)
                combined = combined + name_blend * n_norm

            scores = combined
        else:
            # Fallback heuristic
            scores = df_cand['name_sim_bge'].values * 0.7 + df_cand['name_sim_e5'].values * 0.3

        df_cand['final_score'] = scores
        # Softmax confidence
        exp_s = np.exp(scores - np.max(scores))
        df_cand['confidence'] = exp_s / np.sum(exp_s)

        df_sorted = df_cand.sort_values(by='final_score', ascending=False).reset_index(drop=True)
        return df_sorted.head(top_k), domain_str

def main():
    parser = argparse.ArgumentParser(description="Zero-Inference LLM Router CLI")
    parser.add_argument("query", nargs="?", default=None, help="User prompt to route")
    parser.add_argument("--top_k", type=int, default=5, help="Number of candidate LLMs to recommend")
    args = parser.parse_args()

    router = FastLLMRouter()

    if args.query:
        queries = [args.query]
    else:
        print("\nEntering Interactive Routing Mode. Type 'exit' to quit.\n")
        queries = None

    while True:
        if queries:
            q = queries.pop(0)
        else:
            try:
                q = input("\nEnter query: ")
                if q.strip().lower() in ['exit', 'quit', 'q']:
                    break
                if not q.strip():
                    continue
            except (EOFError, KeyboardInterrupt):
                break

        top_df, domain_str = router.route(q, top_k=args.top_k)

        print("\n" + "="*75)
        print(f"QUERY: {q}")
        print(f"DETECTED INTENT DOMAIN: {domain_str}")
        print("="*75)
        print(f"{'Rank':<6}{'Recommended Candidate LLM':<36}{'Score':<14}{'Confidence'}")
        print("-" * 75)
        for rank, row in top_df.iterrows():
            print(f"{rank+1:<6}{row['llm_id']:<36}{row['final_score']:<14.4f}{row['confidence']*100:.1f}%")
        print("="*75 + "\n")

        if queries is not None and len(queries) == 0:
            break

if __name__ == '__main__':
    main()
