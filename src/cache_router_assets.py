import os
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from router_gemma import extract_discovery_data

def cache_assets():
    cache_path = 'data/router_inference_cache.pkl'
    if os.path.exists(cache_path):
        print(f"Inference cache already exists at {cache_path}!")
        return

    data_dir = 'data/data'
    print("Extracting discovery metadata...")
    disc_queries, disc_stats, llm_global_stats, llm_profiles, _ = extract_discovery_data(
        f"{data_dir}/llm_discovery_metadata.json"
    )
    llm_ids = list(llm_profiles.keys())
    clean_llm_names = [lid.replace('_', ' ') for lid in llm_ids]
    profile_texts = [llm_profiles[lid] for lid in llm_ids]

    print("Building BM25 indices...")
    tokenized_profiles = [doc.lower().split() for doc in profile_texts]
    bm25_profile = BM25Okapi(tokenized_profiles)

    tokenized_names = [n.lower().split() for n in clean_llm_names]
    bm25_name = BM25Okapi(tokenized_names)

    print("Loading transformer models and precomputing offline embeddings...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m_bge = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
    m_bge_large = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
    m_e5 = SentenceTransformer('intfloat/e5-large-v2', device=device)

    # Encode profile bodies
    profile_embs = {
        'bge': m_bge.encode(profile_texts, convert_to_tensor=True, show_progress_bar=False).cpu(),
        'bge_large': m_bge_large.encode(profile_texts, convert_to_tensor=True, show_progress_bar=False).cpu(),
        'e5': m_e5.encode(profile_texts, convert_to_tensor=True, show_progress_bar=False).cpu()
    }

    # Encode category names
    name_bge_embs = m_bge_large.encode(clean_llm_names, convert_to_tensor=True, show_progress_bar=False).cpu()
    name_e5_embs = m_e5.encode(['passage: ' + n for n in clean_llm_names], convert_to_tensor=True, show_progress_bar=False).cpu()

    # Encode discovery queries
    disc_qids = list(disc_queries.keys())
    disc_qtexts = list(disc_queries.values())
    disc_q_embeddings = m_bge.encode(disc_qtexts, convert_to_tensor=True, show_progress_bar=False).cpu()

    cache_bundle = {
        'disc_queries': disc_queries,
        'disc_qids': disc_qids,
        'disc_stats': disc_stats,
        'llm_global_stats': llm_global_stats,
        'llm_ids': llm_ids,
        'clean_llm_names': clean_llm_names,
        'bm25_profile': bm25_profile,
        'bm25_name': bm25_name,
        'profile_embs': profile_embs,
        'name_bge_embs': name_bge_embs,
        'name_e5_embs': name_e5_embs,
        'disc_q_embeddings': disc_q_embeddings
    }

    with open(cache_path, 'wb') as f:
        pickle.dump(cache_bundle, f)
    print(f"Saved complete offline inference cache to {cache_path}!")

if __name__ == '__main__':
    cache_assets()
