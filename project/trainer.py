
import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel, DataCollatorForSeq2Seq

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os

class TrainingArguments:
    def __init__(self, lr, epochs, batch_size):
        self.optim = 'adamw_torch'
        self.eval_strategy = 'epoch'
        self.use_cpu = False
        self.no_cuda = False
        self.learning_rate = lr
        self.num_train_epochs = epochs
        self.per_device_train_batch_size = batch_size
        self.per_device_eval_batch_size = batch_size
        self.output_dir = "."

class ProjectTrainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, dataset, tokenizer):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = dataset["train"]
        self.eval_dataset = dataset["test"]
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        train_loader = DataLoader(self.train_dataset, 
                                  batch_size=args.per_device_train_batch_size,
                                  shuffle=False,
                                  collate_fn=data_collator)
        val_loader = DataLoader(self.eval_dataset, 
                                batch_size=args.per_device_eval_batch_size,
                                shuffle=False,
                                collate_fn=data_collator)
        
        # TODO: Your work here is to implement the training loop.
        self.model.train()
        for epoch in range(args.num_train_epochs):
            step = 0
            for batch in train_loader:
                #       FORWARD PASS:
                
                encoder_input = batch["input_ids"].to(device)
                target_ids = batch["labels"]

                decoder_input = target_ids[:, :-1]
                decoder_input = decoder_input.masked_fill(decoder_input == -100, self.tokenizer.pad_token_id).to(device)
                ground_truth = target_ids[:, 1:].to(device)

                logit_results = self.model(source_lang_ids=encoder_input, target_lang_ids=decoder_input)

                #       compute the loss for the model output and Y
                loss = loss_func(logit_results.reshape(-1, logit_results.size(-1)), ground_truth.reshape(-1))

                if step % 1500 == 0:
                    print(f"At epoch {epoch}, batch {step}, loss = {loss.item():.3f}", flush=True)
                step +=1

                #       BACKWARD PASS AND MODEL UPDATE:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.compute_perplexity(val_loader, loss_func, device) # computes per epoch now

        print(f'\n#Saving to {args.output_dir}.\n#')
        self.model.save_pretrained("trained_model")
    

    def compute_perplexity(self, val_loader, loss_func, device):
        self.model.eval()
        with torch.no_grad():
                total_loss = 0.0
                total_tokens = 0
                pad_id = self.tokenizer.pad_token_id
                for batch in val_loader:

                    encoder_input = batch["input_ids"].to(device)
                    target_ids = batch["labels"]

                    decoder_input = target_ids[:, :-1]
                    decoder_input = decoder_input.masked_fill(decoder_input == -100, self.tokenizer.pad_token_id).to(device)
                    ground_truth = target_ids[:, 1:].to(device)

                    logit_results = self.model(source_lang_ids=encoder_input, target_lang_ids=decoder_input)


                    attn = batch['attention_mask'].to(device)
                    valid = attn[:, 1:]  # mask for Y to exclude padding tokens

                    loss = loss_func(logit_results.reshape(-1, logit_results.size(-1)), ground_truth.reshape(-1))

                    num_valid = valid.sum().item() #since 1 is non-padding token, summing here gives us all non-padding tokens
                    total_loss += loss.item() * num_valid
                    total_tokens += num_valid

                perplexity = float(np.exp(total_loss / total_tokens))
                print(f"#\nPerplexity for the epoch is: {perplexity:.3f} and the loss is {total_loss/len(val_loader):.3f}\n#")
                self.model.train()
