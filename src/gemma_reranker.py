import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
    if prompt is None:
        prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    
    texts = []
    for query, passage in pairs:
        texts.append(f"A: {query}\nB: {passage}\n{prompt}")
        
    # We set padding_side to left for decoder models typically, but for reranker we extract the LAST token logit.
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

def test_gemma_reranker():
    print("Loading Gemma-2B Reranker with device_map='auto'...")
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-gemma')
    model = AutoModelForCausalLM.from_pretrained('BAAI/bge-reranker-v2-gemma', device_map='auto', torch_dtype=torch.float16)
    yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
    model.eval()

    pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda is a bear species endemic to China.']]
    with torch.no_grad():
        inputs = get_inputs(pairs, tokenizer)
        # Move inputs to model device (cuda:0)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        scores = model(**inputs, return_dict=True).logits[:, -1, yes_loc].view(-1, ).float()
        print("Scores:", scores)

if __name__ == "__main__":
    test_gemma_reranker()
