import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def test():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-gemma')
    
    print("Loading model with device_map='auto'...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'BAAI/bge-reranker-v2-gemma', 
        device_map='auto', 
        torch_dtype=torch.float16
    )
    model.eval()
    
    print("Predicting...")
    pairs = [("What is Python?", "Python is a programming language.")]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors='pt')
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logits = model(**inputs, return_dict=True).logits
        scores = logits.view(-1,).float()
        print("Success:", scores)

if __name__ == "__main__":
    test()
