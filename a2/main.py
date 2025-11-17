from utils.runtime_args import get_runtime_args
from a2.run_prediction import predict
from utils.run_train import train
from utils.run_tokenizer import create_tokenizer
from a2.A2_skeleton import (
    A2ModelConfig,
    A2Transformer
)
from utils.tokenizer import A1Tokenizer

import os
file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(file_path)


args = get_runtime_args(script_dir)

if args.run == "predict":
    print("Doing prediction")
    print(predict(A2Transformer, A1Tokenizer, args))

elif args.run == "train":
    print("Initializing training")
    train(args, A2ModelConfig, A2Transformer)

elif args.run == "tokenize":
    print("Building tokenizer")
    create_tokenizer(args)

elif args.run == "all":
    print("Building tokenizer")
    create_tokenizer(args)
    print("Initializing training")
    train(args)
    print("Doing prediction")
    print(predict(args))

else:
    print("Method you are trying to do is not implemented, choose one of the following: \n"
          "- predict \n"
          "- train \n"
          "- tokenize \n"
          "- all"
          )
