
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--run", default="tokenizer")

    # Tokenizer arguments
    parser.add_argument("--vocab-size", default=50000)
    parser.add_argument("--model-max-length", default=1028)
    parser.add_argument("--token-output-dir", default="hf_compatible_trilingual_tokenizer")

    # Dataset arguments
    parser.add_argument("--l1", help="the first language we want to use", default="sv")
    parser.add_argument("--l2", help="the second language we want to use", default="it")
    parser.add_argument("--data-limit", default=None)
    parser.add_argument("--split-size", help="Percentage of test data", default=0.2)
    parser.add_argument("--token-ds-out-path", default="tokenized_datasets/")
    
    # Training arguments
    parser.add_argument("--epochs", default=5)
    parser.add_argument("--lr", default=0.0001)
    parser.add_argument("--batch-size", default=32)
    parser.add_argument("--save-model-dir", default="trained_model")
    parser.add_argument("--load-model-dir", default="trained_model_e5_sv_masking")
    parser.add_argument("--dataset-load-name", default="sv_en_dataset_tokenized")


    # Model config argument
    parser.add_argument("--hidden-size", default=256)
    parser.add_argument("--intermediate-size", default=512)
    parser.add_argument("--num-attention-heads", default=4)
    parser.add_argument("--num-hidden-layers", default=5)
    parser.add_argument("--rms-norm-eps", default=0.001)


    return parser.parse_args()