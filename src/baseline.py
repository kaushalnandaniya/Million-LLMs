import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

def run_baseline():
    data_dir = 'data/data'
    print("Loading official TREC data...")
    if not os.path.exists(f'{data_dir}/llm_discovery_data.json'):
        print(f"Data not found at {data_dir}. Please ensure it's downloaded.")
        return

    # 1. Load Dev Queries (TSV: query_id \t text)
    print("Loading Dev Queries...")
    dev_queries = []
    with open(f'{data_dir}/llm_dev_data.tsv', 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                dev_queries.append({'query_id': parts[0], 'text': parts[1]})
    dev_queries_df = pd.DataFrame(dev_queries)
    
    # 2. Load Discovery Data (JSON: query_id -> {query, responses: {llm_id: response}})
    print("Loading Discovery Data (this might take a moment)...")
    with open(f'{data_dir}/llm_discovery_data.json', 'r') as f:
        discovery_data = json.load(f)

    # 3. Build LLM profiles
    print("Building LLM profiles from past responses...")
    llm_profiles = {}
    for qid, q_data in discovery_data.items():
        responses = q_data.get('responses', {})
        for llm_id, response_text in responses.items():
            if response_text and response_text.lower() != "i don't know":
                if llm_id not in llm_profiles:
                    llm_profiles[llm_id] = []
                # Keep profiles somewhat manageable by sampling if needed, or just append
                llm_profiles[llm_id].append(response_text)
                
    # Combine texts
    for llm_id in llm_profiles:
        # Join max 10 past responses to save memory/time for the baseline
        llm_profiles[llm_id] = " ".join(llm_profiles[llm_id][:10])

    llm_ids = list(llm_profiles.keys())
    profile_texts = list(llm_profiles.values())
    
    print(f"Initializing Sentence Transformer model for {len(llm_ids)} LLMs...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding profiles...")
    profile_embeddings = model.encode(profile_texts, show_progress_bar=True)
    
    print("Ranking LLMs for Dev Queries...")
    results = []
    
    for _, row in dev_queries_df.iterrows():
        qid = row['query_id']
        qtext = row['text']
        
        q_emb = model.encode([qtext])[0]
        sim_scores = cosine_similarity([q_emb], profile_embeddings)[0]
        
        # Sort in descending order
        ranked_indices = np.argsort(sim_scores)[::-1]
        
        for rank, idx in enumerate(ranked_indices):
            # TREC Format: query_id Q0 llm_id rank score run_id
            results.append({
                'query_id': qid,
                'Q0': 'Q0',
                'llm_id': llm_ids[idx],
                'rank': rank + 1,
                'score': sim_scores[idx],
                'run_id': 'baseline_run'
            })
            
    # Save in TREC format
    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    # TREC format is space separated
    results_df.to_csv('results/baseline_run.trec', sep=' ', index=False, header=False)
    
    print("Baseline run complete. Results saved to results/baseline_run.trec")

if __name__ == "__main__":
    run_baseline()
