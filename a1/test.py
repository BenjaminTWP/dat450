from torch.utils.data import DataLoader
from datasets import load_dataset
from A1_skeleton import *
import numpy as np
import torch
import sys


train_file_loc = "/data/courses/2025_dat450_dit247/assignments/a1/train.txt"
val_file_loc = "/data/courses/2025_dat450_dit247/assignments/a1/val.txt"


def test_tokenizer(tokenizer):
    test_texts = ['This is a test.', 'Another test.']
    test = tokenizer(test_texts, return_tensors='pt', padding=True, truncation=True)
    print("# \n Tokenizer sanity check:")
    print(test_texts)
    print(test)
    print("# \n")


def test_dataset():
    print("#\nDataset loading sanity check: \n")
    dataset = load_dataset('text', data_files={'train': train_file_loc, 'val': val_file_loc})
    dataset = dataset.filter(lambda x: x['text'].strip() != '')
    print(f"Length of training dataset is {len(dataset['train'])}")
    print(f"Length of validation dataset is {len(dataset['val'])}")
    
    dl = DataLoader(dataset['train'], batch_size=2, shuffle=False)

    for batch in dl:
        print("First batch from training dataloader (no shuffle): \n")
        print(batch)
        print("#\n")
        break


def test_model():
    print("Model construction sanity check with a vector: \n")
    intgrs = np.random.randint(low=0, high=500, size=100).tolist()
    int_tensor = torch.tensor(intgrs)
    config = A1RNNModelConfig(vocab_size=30000, embedding_size=len(intgrs), hidden_size=2)

    model = A1RNNModel(config)
    result = model.forward(int_tensor)
    print("100 dimensional tensor input to embeedding:\n", result)
    print("Vocab_size of 30000 with the embedding:\n", result.shape)
    print("\n#\n")


def train_model(tokenizer):
    dataset = load_dataset('text', data_files={'train': train_file_loc, 'val': val_file_loc})
    dataset = dataset.filter(lambda x: x['text'].strip() != '')

    tokenizer = tokenizer

    config = A1RNNModelConfig(vocab_size=len(tokenizer),
                                embedding_size=128,
                                hidden_size=64)

    model = A1RNNModel(config)

    args = TrainingArguments(lr=1e-3, epochs=2, batch_size=16)

    a1_trainer = A1Trainer(model, args, dataset['train'], dataset['val'], tokenizer)

    a1_trainer.train()

    return model


def test_some_sentences(model, tokenizer):
    sentences = ["She lives in San", 
                 "The lecture is about to"]

    for sentence in sentences:
        print(f"The 5 best results for following sentence '{sentence}' is: \n")

        encoding = tokenizer([sentence], return_tensors='pt')
        output = model(encoding['input_ids'])
        topk = torch.topk(output[0, -2], k=5)

        for idx, score in zip(topk.indices, topk.values):
            
            print(tokenizer.int_to_str[idx.item()], float(score))
        
        print("\n")


if __name__ == "__main__":
    print("#", flush=True)

    tokenizer = build_tokenizer(train_file_loc, max_voc_size=150000, model_max_length=256)
    tokenizer.save("training")
    tokenizer = A1Tokenizer.from_file("training")

    test_tokenizer(tokenizer)
    test_dataset()
    test_model()

    model = train_model(tokenizer)
    model = A1RNNModel.from_pretrained('trained_model')

    test_some_sentences(model, tokenizer)