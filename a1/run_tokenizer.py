
from A1_skeleton import build_tokenizer
from paths import (
    TOKENIZER,
    TRAIN_FILE,
)

def create_tokenizer(args):
    length = args.modlength
    if length == -1:
        length = None
    obj = build_tokenizer(args.tf, max_voc_size=args.vocsize, model_max_length=length)
    obj.save(TOKENIZER)

