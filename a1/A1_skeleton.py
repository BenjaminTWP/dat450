
import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os
from tqdm import tqdm
import pandas as pd

###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]


def create_vocabulary(tokenized_words, max_voc_size, pad_token, unk_token, bos_token, eos_token):
    cnt = Counter()
    for word in tokenized_words:
        cnt[word] += 1

    #  More words than allowed then keep only the most frequently used ones
    max_voc_size = len(cnt) + 5 if not max_voc_size else max_voc_size
    if len(cnt) > max_voc_size-4:
        cnt = Counter(cnt.most_common(max_voc_size-4))
    
    str_to_int = {pad_token : 0, unk_token : 1, bos_token : 2, eos_token : 3}
    int_to_str = {0 : pad_token, 1: unk_token, 2 : bos_token, 3 : eos_token}
    for i, (word, _) in enumerate(cnt):
        str_to_int[word] = i + 4
        int_to_str[i+4] = word

    return str_to_int, int_to_str


def build_tokenizer(train_file, tokenize_fun=lowercase_tokenizer, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             tokenize_fun:      The function that maps a text to a list of string tokens.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """
    stripped_text = ""
    with open(train_file, "r") as train_file:
        lines = train_file.readlines()
        stripped_lines = [line.strip("\n") for line in lines if line.strip("\n")]
        stripped_text = " ".join(stripped_lines)
    
    tokenized_words = tokenize_fun(stripped_text)

        # If the text is longer than we allow remove the excess words
        # TODO: Should we do this?
        #if len(tokenized_words) > model_max_length:
        #    tokenized_words = tokenized_words[:model_max_length]

    str_to_int, int_to_str = create_vocabulary(tokenized_words, max_voc_size, pad_token, unk_token, bos_token, eos_token)

    return A1Tokenizer(
        str_to_int, 
        int_to_str, 
        model_max_length, 
        str_to_int.get(pad_token),
        str_to_int.get(unk_token),
        bos_token,
        eos_token
    )

    
    # TODO: build the vocabulary, possibly truncating it to max_voc_size if that is specified.
    # Then return a tokenizer object (implemented below).
    ...

class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, str_to_int, int_to_str, model_max_length, pad_id, unk_id, bos_token, eos_token):
        # TODO: store all values you need in order to implement __call__ below.
        self.str_to_int = str_to_int
        self.int_to_str = int_to_str
        self.pad_token_id = pad_id     # Compulsory attribute.
        self.model_max_length = model_max_length # Needed for truncation.
        self.unk_id = unk_id
        self.bos_token = bos_token
        self.eos_token = eos_token

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.
           
           Args:
             texts:           The texts to tokenize.
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')
        
        # TODO: Your work here is to split the texts into words and map them to integer values.
        # 
        # - If `truncation` is set to True, the length of the encoded sequences should be 
        #   at most self.model_max_length.
        # - If `padding` is set to True, then all the integer-encoded sequences should be of the
        #   same length. That is: the shorter sequences should be "padded" by adding dummy padding
        #   tokens on the right side.
        # - If `return_tensors` is undefined, then the returned `input_ids` should be a list of lists.
        #   Otherwise, if `return_tensors` is 'pt', then `input_ids` should be a PyTorch 2D tensor.

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an 
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.

        # This is predefined, so we get a list of texts, something like  [['This is a test.', 'Another test.']]
        tokenizations = []
        attention_mask = []
        longest_text = 0 
        for text in texts:
            text_mapping = []
            text = text.strip("\n")
            text_length = len(text)
            if truncation and self.model_max_length and text_length > self.model_max_length:
                text = text[:self.model_max_length]
            
            tokenized_words = lowercase_tokenizer(text)
            tokenized_words.insert(0, self.bos_token)
            tokenized_words.append(self.eos_token)
            longest_text = longest_text if longest_text > len(tokenized_words) else len(tokenized_words)
            for word in tokenized_words:
                word_int = self.str_to_int.get(word)
                if not word_int:
                    word_int = self.unk_id

                text_mapping.append(word_int)

            tokenizations.append(text_mapping)
            attention_mask.append([1] * len(text_mapping))

   
        if padding:
            for i  in range(len(tokenizations)):
                text_tokenization = tokenizations[i]
                diff = longest_text - len(text_tokenization)
                if diff > 0:
                    text_tokenization.extend([self.pad_token_id] * diff)
                    attention_mask[i].extend([0] * diff)
        
        if return_tensors == 'pt':
            return BatchEncoding({
                'attention_mask' : torch.tensor(attention_mask),
                'input_ids': torch.tensor(tokenizations)
            }) 
        
        
        return BatchEncoding({
                'attention_mask' : attention_mask,
                'input_ids': tokenizations
            }) 

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.str_to_int)
    
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
   

###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=None, embedding_size=None, hidden_size=None,
                 num_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.embedding_size
        )
        self.rnn = nn.LSTM(
            input_size=self.config.embedding_size,
            hidden_size=self.config.hidden_size, 
            num_layers=self.config.num_layers,
            batch_first=True
        )
        self.unembedding = nn.Sequential(
            nn.Linear(
                in_features=self.rnn.hidden_size,
                out_features=self.config.vocab_size
            ),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, X):
        """The forward pass of the RNN-based language model.
        
           Args:
             X:  The input tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
        """
        embedded = self.embedding(X)
        rnn_out, _ = self.rnn(embedded)
        out = self.unembedding(rnn_out)
        return out


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size: 
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.

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
        train_loader = DataLoader(self.train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
        val_loader = DataLoader(self.eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)
        
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

        def get_model_loss(batch):
            batch_encoding = self.tokenizer(batch.get("text"), return_tensors='pt', padding=True, truncation=True)
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
            val_loss = np.zeros(2)
            self.model.train()
            train_batch_progress_bar = tqdm(
                train_loader, 
                desc=f"Training - Batch progress, epoch {i}", 
                ncols=100
            )
            for batch in train_batch_progress_bar:
                loss = get_model_loss(batch)
                train_loss[0] += len(batch.get("text"))
                train_loss[1] += loss 
                train_batch_progress_bar.set_postfix(
                    training_loss=train_loss[1].item()/train_loss[0]
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
                        val_loss[0] += len(batch.get("text"))
                        val_loss[1] += get_model_loss(batch)
                        val_batch_progress_bar.set_postfix(
                            validation_loss=val_loss[1].item()/val_loss[0] 
                        ), 

            train_avrg = train_loss[1].item()/train_loss[0]
            val_avrg = val_loss[1].item()/val_loss[0]

            # TODO: Check perplexity formula, divide by length of loader aka number of batches or number of samples
            # REmove padding when calculating perplexity. Currently we also count the padding, this we shouldn't do
            perplexity = np.exp(val_avrg)
            losses = pd.concat([
                pd.DataFrame(
                    [[i, train_loss[1].item(), val_loss[1].item(), train_avrg, val_avrg, perplexity]], 
                    columns=losses.columns
                ), 
                losses
            ])

            print(
                "\n---------------------------------------------------\n"
                f"Epoch {i+1}/{args.num_train_epochs} finished!\n"
                f"Final losses for epoch was:\n"
                f" * Total training: {train_loss[1]}\n"
                f" * Total validation: {val_loss[1]}\n"
                f" - Avrg training: {train_avrg:.4f}\n"
                f" - Avrg validation: {val_avrg:.4f}\n"
                f" ° Perplexity validation {perplexity}\n"
                "---------------------------------------------------\n"
            )

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)
        losses.to_csv(args.output_dir + "/losses.csv", index=False)



