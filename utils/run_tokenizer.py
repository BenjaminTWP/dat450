
from utils.tokenizer import build_tokenizer

def create_tokenizer(args):
    length = args.modlength
    if length == -1:
        length = None
    obj = build_tokenizer(args.tf, max_voc_size=args.vocab_size, model_max_length=length)
    obj.save(args.tokenizer_file)

