
import torch, nltk, pickle
nltk.download('punkt')
nltk.download('punkt_tab')

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd


class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
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
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if not self.args.no_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.learning_rate, 
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay 
        )

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.per_device_train_batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False
        )
        
        # TODO: Your work here is to implement the training loop.
        #       
        # for each training epoch (use args.num_train_epochs here):
        #   for each batch B in the training set:
        #
        #       PREPROCESSING AND FORWARD PASS:
        #       input_ids = apply your tokenizer to B
	    #       X = all columns in input_ids except the last one
	    #       Y = all columns in input_ids except the first one
	    #       put X and Y onto the GPU (or whatever device you use)
        #       apply the model to X
        #   	compute the loss for the model output and Y
        #
        #       BACKWARD PASS AND MODEL UPDATE:
        #       optimizer.zero_grad()
        #       loss.backward()
        #       optimizer.step()

        def get_model_loss(batch_encoding):
            input_ids = batch_encoding.get("input_ids")
            X = input_ids[:, :-1]
            Y = input_ids[:, 1:]
            X = X.to(device)
            Y = Y.to(device)
            preds = self.model(X)
            return loss_func(preds.view(-1, preds.shape[-1]), Y.reshape(-1))
        
        losses = pd.DataFrame(columns=["Epoch", "Total training loss", "Total val loss", "Avrg training loss", "Avrg val loss", "Perplexity"])
        for i in range(args.num_train_epochs):
            train_loss = np.zeros(2)
            val_loss = np.zeros(4)
            self.model.train()
            train_batch_progress_bar = tqdm(
                train_loader, 
                desc=f"Training - Batch progress, epoch {i}", 
                ncols=100
            )
            for batch in train_batch_progress_bar:
                batch_encoding = self.tokenizer(batch.get("text"), return_tensors='pt', padding=True, truncation=True)
                loss = get_model_loss(batch_encoding)
                train_loss[0] += batch_encoding['attention_mask'].sum()
                train_loss[1] += loss.item()
                train_batch_progress_bar.set_postfix(
                    training_loss=loss.item()
                ), 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if args.eval_strategy == 'epoch':
                self.model.eval()
                val_batch_progress_bar = tqdm(
                    val_loader, 
                    desc=f"Validation - Batch progress, epoch {i}", 
                    ncols=100
                )
                with torch.no_grad():
                    for batch in val_batch_progress_bar:
                        batch_encoding = self.tokenizer(batch.get("text"), return_tensors='pt', padding=True, truncation=True)
                        valid = batch_encoding['attention_mask'].to(device)
                        # Calculate how many batches we recived for our training examples.
                        nr_batches = valid[:, 0].sum() 
                        nr_valid_tokens= valid.sum()
                        loss = get_model_loss(batch_encoding)

                        # loss returned as avrg over batch so has to "un_average" it by mult with the number of batches
                        #val_loss[2] += loss.item()
                        
                        # These are used for calculating the average etc
                        val_loss[3] += loss.item() * nr_batches
                        val_loss[2] += nr_batches
                        val_loss[1] += loss.item()
                        val_loss[0] += nr_valid_tokens
                        val_batch_progress_bar.set_postfix(
                            validation_loss=loss.item()
                        ), 


            train_avrg = train_loss[1].item()/train_loss[0].item()
            val_avrg = val_loss[1]/val_loss[0].item()
            
            print(train_loss)
            print(val_loss)

            # TODO: check this but should be accurate
            perplexity = np.exp(val_loss[3]/val_loss[2])
            losses = pd.concat([
                pd.DataFrame(
                    [[i, train_loss[1], val_loss[1], train_avrg, val_avrg, perplexity]], 
                    columns=losses.columns
                ), 
                losses
            ])

            # TODO: THe total training will be the sum of the batch averages so this is kinda wrong
            # TODO: Here the output is the average over the average etc, change it
            print(
                "\n---------------------------------------------------\n"
                f"Epoch {i+1}/{args.num_train_epochs} finished!\n"
                f"Final losses for epoch was:\n"
                f" * Summed average training: {train_loss[1]}\n"
                f" * Summed average validation: {val_loss[1]}\n"
                #f" - Averaged Avrg training: {train_avrg:.4f}\n"
                #f" - Averaged Avrg validation: {val_avrg:.4f}\n"
                f" ° Perplexity validation {perplexity}\n"
                "---------------------------------------------------\n"
            )

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)
        losses.to_csv(args.output_dir + "/losses.csv", index=False)

