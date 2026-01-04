
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--run", type=str, default="tokenizer")

    # Tokenizer arguments
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--model-max-length", type=int, default=1028)
    parser.add_argument("--token-output-dir", type=str, default="my_tokenizer")

    # Dataset arguments
    parser.add_argument("--l1", help="the first language we want to use", type=str, default="sv")
    parser.add_argument("--l2", help="the second language we want to use", type=str, default="it")
    parser.add_argument("--data-limit", type=int, default=1000000)
    parser.add_argument("--split-size", help="Percentage of test data", type=float, default=0.2)
    parser.add_argument("--token-ds-out-path", type=str, default="tokenized_datasets/")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-model-dir", type=str, default="trained_model")
    parser.add_argument("--load-model-dir", default=None)
    parser.add_argument("--dataset-load-name", type=str, default="sv_en_dataset_tokenized")


    # Model config argument
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--num-hidden-layers", type=int, default=5)
    parser.add_argument("--rms-norm-eps", type=float, default=0.001)

    # Load Helsinki model
    parser.add_argument("--is-helsinki", action='store_true')
    parser.add_argument("--helsinki-model", type=str, default="Helsinki-NLP/opus-mt-sv-en")


    # Generation arguments
    parser.add_argument("--gen-len", type=int, default=200)

    return parser.parse_args()