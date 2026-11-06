import torch
from sentence_transformers.cross_encoder import CrossEncoder

def test():
    print("Loading BAAI/bge-reranker-v2-gemma with device_map='auto'...")
    try:
        ce = CrossEncoder('BAAI/bge-reranker-v2-gemma', model_kwargs={"device_map": "auto", "torch_dtype": torch.float16})
        scores = ce.predict([("What is Python?", "Python is a programming language.")], batch_size=1)
        print("Success:", scores)
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    test()
