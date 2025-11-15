from a1.A1_skeleton import A1RNNModel
from utils.tokenizer import A1Tokenizer
import torch

def predict(args):
    model = A1RNNModel.from_pretrained(args.output_dir)
    tokenizer = A1Tokenizer.from_file(args.tokenizer_file)

    example_text = [input("Enter a sentence in which you want to predict the next word for: \n- ").strip() or "This is the default string"]

    batch_encoding = tokenizer(example_text, return_tensors='pt', padding=False, truncation=False)

    # TODO: Either one can slice away the end part after we do the tokanization or we do it during predictions
    # Shouldn't have to take any padding into account
    remove_eos_tensor = batch_encoding.get("input_ids")[:, :-1]
    preds = model(remove_eos_tensor)


    probs, indices = torch.topk(preds, k=args.npreds, dim=-1)

    best_preds = []
    for i, pred in enumerate(indices[:, -1, :].flatten()):
        best_preds.append((tokenizer.int_to_str.get(pred.item()), probs[:, -1, i].item()))

    # TODO: Look at why we have high certainty for unknown, prob due to small vocab
    print(best_preds)
