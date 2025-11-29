import copy

import torch

# -----------------------------------------------------------------------------
# STUDENT TODOs: Prompt + Tokenization utilities
# -----------------------------------------------------------------------------
# Fill in the functions below.
#   * Turn raw Alpaca examples into prompts (`build_prompt`)
#   * Convert prompt/answer pairs into causal-LM tensors (`tokenize_helper`)
#   * Batch variable-length tensors safely during training (`create_data_collator`)
#
# Hints are provided inline. You can look at the instructor utilities for
# reference after attempting the implementation on your own.
# -----------------------------------------------------------------------------


def build_prompt(example, prompt_no_input, prompt_with_input):
    """
    Build an Alpaca-style prompt and keep the gold output as 'answer'.

    Args:
        example: Dict with keys like 'instruction', 'input', 'output'.
        prompt_no_input: Template for instructions without extra context.
        prompt_with_input: Template for instructions that provide an input.
    Example:
        >>> build_prompt(
        ...     {"instruction": "Do X", "input": "Extra info", "output": "Result"},
        ...     prompt_no_input="... {instruction} ...",
        ...     prompt_with_input="... {instruction} ... {input} ...",
        ... )
        {"prompt": "... Do X ... Extra info ...", "answer": "Result"}
    Returns:
        Dict with the text prompt under "prompt" and the gold answer under "answer".
    """
    instruction = example["instruction"]
    inp = example.get("input", "") or ""
    output = example["output"]

    # TODO[student]:
    #   1. Read out instruction/input/output from the example (be robust to empty input).
    #   2. Pick the correct template depending on whether "input" contains text.
    #   3. Format the template and return {"prompt": prompt_text, "answer": output_text}.
    
    #print(example)
    #print(f"This is inp {inp}, adadwda")
    #response_prompt = 
    x = prompt_no_input.replace("{instruction}", instruction)
    if inp:
        x = prompt_with_input.replace("{instruction}", instruction)
        x = x.replace("{input}", inp)

    tmp = {
        "prompt" : x,
        "answer" : output,
    } 
    
    return dict(tmp)


def tokenize_helper(batch, tokenizer, max_length):
    """
    Tokenize prompt+answer pairs for causal LM fine-tuning.

    Requirements:
      * Concatenate prompt IDs with answer IDs (answer prefixed with a space).
      * Truncate to `max_length`.
      * Build labels that ignore prompt tokens (set them to -100) and supervise answer tokens.

    Example:
        >>> tokenize_helper(
        ...     {"prompt": "Prompt", "answer": "Answer"},
        ...     tokenizer=my_tokenizer,
        ...     max_length=32,
        ... )
        {"input_ids": [...], "attention_mask": [...], "labels": [...]}

    Returns:
      dict(input_ids=list[int], attention_mask=list[int], labels=list[int])
    """
    # TODO[student]:
    #   1. Tokenize the prompt (no special tokens) and the answer separately.
    #   2. Concatenate, truncate, and create an attention mask of 1s.
    #   3. Build labels using -100 for the prompt span and answer token IDs afterward.
    #      (Hint: copy answer IDs so truncation does not mutate the tokenizer output.)


    prompt_tokenized = tokenizer(batch.get("prompt"), add_special_tokens=False)
    answer_tokenized = tokenizer(" " + batch.get("answer"), add_special_tokens=False)


    prompt_token_length = len(prompt_tokenized.get("input_ids"))
    prompt_labels = [-100] * prompt_token_length
    prompt_labels.extend(answer_tokenized.get("input_ids"))

    new_input_ids = prompt_tokenized.get("input_ids", []) + answer_tokenized.get("input_ids", [])
    new_attention_mask = prompt_tokenized.get("attention_mask", []) + answer_tokenized.get("attention_mask", [])

    x = {
        "input_ids": new_input_ids,
        "attention_mask" : new_attention_mask,
        "labels" : prompt_labels
    }

    if  prompt_token_length >= max_length:
        for key, value in x.items():
            x[key] = value[:max_length] 

    else:
        diff = max_length - prompt_token_length
        x["input_ids"] = x.get("input_ids") + [tokenizer.pad_token_id] * diff
        x["attention_mask"] = x.get("attention_mask") + [0] * diff
        x["labels"] = x.get("labels") + [-100] * diff

    
    return x


def create_data_collator(tokenizer):
    """
    Create a custom collate function for causal language modeling.

    The collator should dynamically right-pad:
      * input_ids with tokenizer.pad_token_id
      * attention_mask with 0
      * labels with -100
    to the maximum sequence length within each batch.

    Example:
        >>> collate = create_data_collator(tokenizer)
        >>> collate([{"input_ids": [1], "attention_mask": [1], "labels": [5]}])
        {"input_ids": tensor(...), "attention_mask": tensor(...), "labels": tensor(...)}

    """
    # TODO[student]:
    #   1. Inside `data_collator(batch)`, convert each list of ids into torch tensors.
    #   2. Compute the max length in the batch.
    #   3. Pad every tensor on the right to that length (use torch.full for padding values).
    #   4. Stack the padded tensors into a batch dict and return it.
    #
    # Suggested structure:
    #   def data_collator(batch):
    #       ...
    #       return {
    #           "input_ids": ...,
    #           "attention_mask": ...,
    #           "labels": ...,
    #       }
    #   return data_collator

    def data_collator(batch):
        input_ids_list = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
        attention_masks_list = [torch.tensor(example["attention_mask"], dtype=torch.long) for example in batch]
        labels_list = [torch.tensor(example["labels"], dtype=torch.long) for example in batch]

        # Find max length in this batch
        max_len = max(len(input_list) for input_list in input_ids_list) 
        # print(max_len)

        # Helper pad function: right-pad to max_len
        def pad_to_max(x_list, pad_value):
            for i, x_tensor in enumerate(x_list):
                diff = max_len - len(x_tensor)
                if diff > 0:
                    fill_tensor = torch.full(size=(diff,), fill_value=pad_value)
                    # print(x_tensor.size())
                    # print(fill_tensor.size())
                    x_list[i] = torch.cat((x_tensor, fill_tensor), dim=0)

            #x_tensor == 100277 
            return x_list
        # Use tokenizer.pad_token_id for inputs, 0 for attention_mask, -100 for labels
        pad_id = tokenizer.pad_token_id

        batch_input_ids = pad_to_max(input_ids_list, pad_value=pad_id)
        batch_attention_mask = pad_to_max(attention_masks_list, pad_value=0)
        batch_labels = pad_to_max(labels_list, pad_value=-100)

        # print(batch_input_ids)
        # print(batch_attention_mask)
        # print(batch_labels)

        # raise Exception("Heeaf")

        batch = {
            "input_ids": torch.stack(batch_input_ids, dim=0) ,
            "attention_mask": torch.stack(batch_attention_mask, dim=0),
            "labels": torch.stack(batch_labels, dim=0),
        }
        return batch

    return data_collator

    raise NotImplementedError("Implement the causal-LM data collator.")
