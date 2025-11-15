import torch
from torch.distributions import Categorical

def predict(model, tokenizer, args):
    model = model.from_pretrained(args.output_dir)
    tokenizer = tokenizer.from_file(args.tokenizer_file)

    prompt = input("Enter a sentence in which you want to predict the next word for: \n- ").strip() or "She lives in"
    
    print(prompt, end="")
    for i in range(args.generation_max_length):
        encoding = tokenizer([prompt], return_tensors='pt', padding=False, truncation=False)
        logits = model(encoding.input_ids)

        # Since EOS is the last token we have to check the second to last token
        last_logits = logits[:, -2, :]
        scaled_logits = torch.exp(torch.multiply(last_logits, (1-args.temperature)))


        (values, tokens) = torch.topk(scaled_logits, k=args.npreds)

        possible_next_token = [tokenizer.int_to_str.get(key) for key in tokens.squeeze().tolist()]

        distr = Categorical(logits=values)
        sampled = distr.sample()
        chosen_token = possible_next_token[sampled] 
        if chosen_token == tokenizer.eos_token:
            break

        prompt += f" {chosen_token}"
        print(" " + chosen_token, end="")