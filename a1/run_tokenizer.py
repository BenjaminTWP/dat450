
from A1_skeleton import build_tokenizer
from paths import (
    TOKENIZER,
    TRAIN_FILE,
)

def create_tokenizer(args):
    length = args.modlength
    if length == -1:
        length = None
    print(TRAIN_FILE)
    obj = build_tokenizer(TRAIN_FILE, max_voc_size=args.vocsize, model_max_length=length)
    obj.save(TOKENIZER)

