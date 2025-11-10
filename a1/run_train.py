from datasets import load_dataset
from torch.utils.data import Subset
from A1_skeleton import (
    A1RNNModel, 
    A1RNNModelConfig,
    A1Tokenizer,
    A1Trainer
)
from transformers import TrainingArguments
from paths import (
    TRAINER_OUTPUT,
    TOKENIZER
)

def train(args):
    dataset = load_dataset('text', data_files={'train': args.tf, 'val': args.vf})
    dataset = dataset.filter(lambda x: x['text'].strip() != '')

    if args.dl:
        for sec in ['train', 'val']:
            dataset[sec] = Subset(dataset[sec], range(args.dlt))

    tokenizer = A1Tokenizer.from_file(TOKENIZER)

    training_args = TrainingArguments(
        optim="adamw_torch", 
        eval_strategy="epoch",
        learning_rate=args.lr,
        num_train_epochs=args.eps,
        per_device_train_batch_size=args.tbs,
        per_device_eval_batch_size=args.vbs,
        output_dir=TRAINER_OUTPUT
    )

    config = A1RNNModelConfig(
        vocab_size=len(tokenizer),
        embedding_size=args.ebs,
        hidden_size=args.hdn,
        num_layers=args.nlayers
    )

    model = A1RNNModel(config)

    trainer = A1Trainer(
        model, 
        training_args,
        dataset['train'],
        dataset['val'],
        tokenizer
    )

    trainer.train()
