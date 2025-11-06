from runtime_args import get_runtime_args
from run_prediction import predict
from run_train import train
from run_tokenizer import create_tokenizer

args = get_runtime_args()

if args.run == "predict":
    print("Doing prediction")
    predict(args)

elif args.run == "train":
    print("Initializing training")
    train(args)

elif args.run == "tokenize":
    print("Building tokenizer")
    create_tokenizer(args)

elif args.run == "all":
    print("Building tokenizer")
    create_tokenizer(args)
    print("Initializing training")
    train(args)
    print("Doing prediction")
    predict(args)

else:
    print("Method you are trying to do is not implemented, choose one of the following: \n"
          "- predict \n"
          "- train \n"
          "- tokenize \n"
          "- all"
          )
