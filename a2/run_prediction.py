import torch
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM

def predict(model, tokenizer, args, prompt=None):
    prompt = prompt if prompt else args.prompt
    model = model.from_pretrained(args.output_dir)
    tokenizer = tokenizer.from_file(args.tokenizer_file)

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
        
    return prompt
        


def predict_olmo(args, prompt=None):
    prompt = prompt if prompt else args.prompt
    tokenizer = AutoTokenizer.from_pretrained(args.olmo_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.olmo_dir, local_files_only=True)

    for i in range(args.generation_max_length):
        encoding = tokenizer([prompt], return_tensors='pt', padding=False, truncation=False)

        casual_output = model(encoding.input_ids)

        logits = casual_output.logits

        last_logits = logits[:, -1, :]
        scaled_logits = torch.exp(torch.multiply(last_logits, (1-args.temperature)))


        (values, tokens) = torch.topk(scaled_logits, k=args.npreds)

        distr = Categorical(logits=values)
        sampled = distr.sample()
        chosen_token_id = tokens[sampled] 
        if chosen_token_id == tokenizer.eos_token_id:
            break

        chosen_token = tokenizer.decode(chosen_token_id)

        prompt += f" {chosen_token}"

    return prompt