import torch
from sentence_transformers import SentenceTransformer

def test():
    print("Loading Alibaba-NLP/gte-Qwen2-7B-instruct with device_map='auto'...")
    try:
        model = SentenceTransformer(
            "Alibaba-NLP/gte-Qwen2-7B-instruct", 
            model_kwargs={"device_map": "auto", "torch_dtype": torch.float16},
            trust_remote_code=True
        )
        emb = model.encode(["What is Python?", "Python is a programming language."])
        print("Success, shape:", emb.shape)
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    test()
