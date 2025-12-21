import torch

def translate_tokens(model, encoding, tokenizer, device, max_length=50):

    target_lang_ids = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(device)
    
    source_lang_ids = torch.tensor(encoding['input_ids']).unsqueeze(0).to(device)

    attention_mask = torch.tensor(encoding['attention_mask']).unsqueeze(0).to(device)
    
    for _ in range(max_length):

        logits = model(source_lang_ids, attention_mask, target_lang_ids).squeeze(0)

        next_token = torch.argmax(logits, dim=1)[-1].unsqueeze(0).unsqueeze(0)

        target_lang_ids = torch.cat((target_lang_ids, next_token), dim=1)
       
        if next_token == tokenizer.eos_token_id:
            break


    return tokenizer.decode(target_lang_ids.squeeze(0), skip_special_tokens=True)


def translate_sentence(model, prompt, tokenizer, device, max_length=50):

    encoding = tokenizer(prompt)

    return translate_tokens(model, encoding, tokenizer, device, max_length=max_length)

def remove_from_batch(old_tensor, batch_ind):
    return torch.cat((old_tensor[:batch_ind, :], old_tensor[batch_ind+1:, :]), dim=0)

def translate_tokens_batch(model, source_lang_tokens, attention_mask, tokenizer, device, max_length=50):

    batch_size = len(source_lang_tokens)
    
    translated_tokens = torch.full((batch_size, max_length), tokenizer.pad_token_id)

    finished = [False] * batch_size

    target_lang_ids = torch.full((batch_size, 1), tokenizer.bos_token_id).to(device)
    
    for _ in range(max_length):

        logits = model(source_lang_tokens, attention_mask, target_lang_ids)

        next_token = torch.argmax(logits, dim=2)[:, -1].unsqueeze(1)

        target_lang_ids = torch.cat((target_lang_ids, next_token), dim=1)

        completed_inds = torch.where(next_token == tokenizer.eos_token_id)[0].tolist()
        to_remove_inds = completed_inds.copy()

        number_of_skips = 0
        for i in range(batch_size):

            if len(completed_inds) == 0:
                break
            ind = completed_inds[0]

            if finished[i]: # spot already filled
                continue
            else:
                if number_of_skips == ind:
                    finished[i] = True
                    target_len = len(target_lang_ids[ind])
                    translated_tokens[i, :min(target_len, 50)] = target_lang_ids[ind, :min(target_len, 50)]
                    completed_inds.pop(0)
                number_of_skips += 1

        for ind in reversed(to_remove_inds):
            target_lang_ids = remove_from_batch(target_lang_ids, ind)
            source_lang_tokens = remove_from_batch(source_lang_tokens, ind)
            attention_mask = remove_from_batch(attention_mask, ind)

        if len(target_lang_ids) == 0:
            break 

    if len(target_lang_ids) != 0:
        for i in range(batch_size):
            if not finished[i]:
                target_len = len(target_lang_ids[0])
                translated_tokens[i, :min(target_len, 50)] = target_lang_ids[0, :min(target_len, 50)]
                target_lang_ids = remove_from_batch(target_lang_ids, 0)
                source_lang_tokens = remove_from_batch(source_lang_tokens, 0)
                attention_mask = remove_from_batch(attention_mask, 0)
                


    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

