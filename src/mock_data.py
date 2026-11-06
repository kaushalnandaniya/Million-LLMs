import pandas as pd
import numpy as np
import os
import random

def generate_mock_data():
    os.makedirs('data', exist_ok=True)
    
    # 1. Generate LLM Metadata
    print("Generating LLM metadata...")
    llms = []
    for i in range(1, 101): # 100 mock LLMs
        llms.append({
            'llm_id': f'llm_{i}',
            'parameter_count_b': random.choice([7, 13, 70, 100, 400]),
            'domain_specialization': random.choice(['general', 'code', 'medical', 'finance', 'math']),
            'response_verbosity': random.choice(['concise', 'detailed', 'verbose'])
        })
    df_llms = pd.DataFrame(llms)
    df_llms.to_csv('data/llm_metadata.csv', index=False)
    
    # 2. Generate Discovery Set (Past queries and responses)
    print("Generating Discovery set (past responses)...")
    discovery_data = []
    topics = ['how to write a python script', 'what is the capital of france', 
              'diagnose a headache', 'invest in stocks', 'solve calculus limit']
    
    for llm in llms:
        # Each LLM has answered a few past queries
        for _ in range(5):
            q = random.choice(topics)
            # Mock a response string
            resp = f"Here is an answer about {q.split()[-1]}. " + "blah " * random.randint(5, 20)
            discovery_data.append({
                'llm_id': llm['llm_id'],
                'discovery_query': q,
                'response': resp
            })
            
    df_discovery = pd.DataFrame(discovery_data)
    df_discovery.to_csv('data/discovery_set.csv', index=False)
    
    # 3. Generate Dev Set (Test queries and ground truth rankings)
    print("Generating Dev set (queries and qrels)...")
    dev_queries = [
        {'query_id': 'q1', 'text': 'write a fast api server in python'},
        {'query_id': 'q2', 'text': 'symptoms of flu vs cold'},
        {'query_id': 'q3', 'text': 'best mutual funds 2026'},
    ]
    df_dev_queries = pd.DataFrame(dev_queries)
    df_dev_queries.to_csv('data/dev_queries.csv', index=False)
    
    # Qrels (Ground truth expertise scores 0-3)
    qrels = []
    for q in dev_queries:
        for llm in llms:
            # Random expertise score for mock data
            score = random.choices([0, 1, 2, 3], weights=[0.6, 0.2, 0.1, 0.1])[0]
            if score > 0:
                qrels.append({
                    'query_id': q['query_id'],
                    'llm_id': llm['llm_id'],
                    'expertise_score': score
                })
    df_qrels = pd.DataFrame(qrels)
    df_qrels.to_csv('data/dev_qrels.csv', index=False)
    print("Mock data generated in data/ directory.")

if __name__ == "__main__":
    generate_mock_data()
