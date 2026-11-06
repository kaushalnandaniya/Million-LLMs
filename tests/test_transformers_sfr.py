import torch
from transformers import AutoTokenizer, AutoModel

def test():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
    
    print("Loading model...")
    model = AutoModel.from_pretrained(
        'Salesforce/SFR-Embedding-Mistral', 
        device_map='balanced',
        max_memory={0: "8GB", 1: "8GB"},
        torch_dtype=torch.float16
    )
    model.eval()
    
    print(f"Model loaded. Devices: GPU 0: {torch.cuda.memory_allocated(0)/(1024**3):.2f}GB, GPU 1: {torch.cuda.memory_allocated(1)/(1024**3):.2f}GB")
    
    print("Predicting...")
    pairs = ["What is Python?", "Python is a programming language."]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors='pt')
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model(**inputs)
        # Use mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        print("Success, shape:", emb.shape)

if __name__ == "__main__":
    test()
