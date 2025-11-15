from datasets import load_dataset
from torch.utils.data import Subset
from utils.tokenizer import A1Tokenizer
from utils.trainer import A1Trainer
from utils.args_safe_insert import safe_create_class
from transformers import TrainingArguments


def train(args, configClass, modelClass):
    dataset = load_dataset('text', data_files={'train': args.tf, 'val': args.vf})
    dataset = dataset.filter(lambda x: x['text'].strip() != '')

    if args.use_data_limit:
        for sec in ['train', 'val']:
            dataset[sec] = Subset(dataset[sec], range(args.data_limit_threshold))


    tokenizer = A1Tokenizer.from_file(args.tokenizer_file)
    

    training_args = safe_create_class(TrainingArguments, args)

    args.vocab_size = len(tokenizer)
    config = safe_create_class(configClass, args)

    model = modelClass(config)

    trainer = A1Trainer(
        model, 
        training_args,
        dataset['train'],
        dataset['val'],
        tokenizer
    )

    trainer.train()
