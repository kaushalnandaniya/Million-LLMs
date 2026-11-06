import torch
from sentence_transformers import SentenceTransformer

def test():
    print("Loading Salesforce/SFR-Embedding-Mistral with device_map='auto'...")
    try:
        model = SentenceTransformer(
            "Salesforce/SFR-Embedding-Mistral", 
            model_kwargs={
                "device_map": "balanced", 
                "torch_dtype": torch.float16,
                "max_memory": {0: "9GB", 1: "9GB"}
            }
        )
        emb = model.encode(["What is Python?", "Python is a programming language."])
        print("Success, shape:", emb.shape)
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    test()
