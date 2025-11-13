
import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os

###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

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

    # TODO: build the vocabulary, possibly truncating it to max_voc_size if that is specified.
    # Then return a tokenizer object (implemented below).
    str_to_int = {pad_token: 0, unk_token: 1, bos_token: 2, eos_token:3}
    int_to_str = {0: pad_token, 1: unk_token, 2: bos_token, 3: eos_token}
    voc_len = 4 # current length
    word_counter = Counter()

    with open(train_file) as file:
        for paragraph in file:
            tokens = tokenize_fun(paragraph)
            if model_max_length:
                tokens = tokens[:model_max_length]

            word_counter.update(tokens) #just inserts and counts the words in the list

    most_common_tokens = []
    if max_voc_size:
        max_other_tokens = max_voc_size - len(str_to_int)
        most_common_tokens = [token for token, count in word_counter.most_common(max_other_tokens)]
    else:
        most_common_tokens = list(word_counter.keys())

    for token in most_common_tokens:
        str_to_int[token] = voc_len
        int_to_str[voc_len] = token
        voc_len += 1

    #print("#\n5 most common words: ", word_counter.most_common(5))
    #print("5 least common words: ", word_counter.most_common()[-5:])
    #rint("Dict of 'the' should inversly map back to 'the': ", int_to_str[str_to_int["the"]]) 
    #print("Dict of 'person' should inversly map back to 'person': ", int_to_str[str_to_int["person"]]) 
    #print(f"Size of vocabulary is {len(str_to_int)} and specified max is {max_voc_size}\n#")

    return A1Tokenizer(str_to_int, int_to_str, {'pad_token': pad_token,
                                                'unk_token': unk_token, 
                                                'bos_token': bos_token, 
                                                'eos_token': eos_token}, model_max_length)


class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, str_to_int, int_to_str, special_tokens, model_max_length):
        # TODO: store all values you need in order to implement __call__ below.
        self.pad_token_id = str_to_int[special_tokens['pad_token']]   # Compulsory attribute.
        self.model_max_length = model_max_length # Needed for truncation.
        self.str_to_int = str_to_int
        self.int_to_str = int_to_str 
        self.special_tokens = special_tokens

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
        # - Encoded sequences should start with the beginning-of-sequence dummy; non-truncated
        #   sequences should end with the end-of-sequence dummy; out-of-vocabulary tokens should
        #   be encoded with the 'unknown' dummy.
        # - If `padding` is set to True, then all the integer-encoded sequences should be of the
        #   same length. That is: the shorter sequences should be "padded" by adding dummy padding
        #   tokens on the right side.
        # - If `return_tensors` is undefined, then the returned `input_ids` should be a list of lists.
        #   Otherwise, if `return_tensors` is 'pt', then `input_ids` should be a PyTorch 2D tensor.

        max_seq = 0
        encodings = []

        for text in texts:
            tokens = lowercase_tokenizer(text)
            one_sequence = [self.str_to_int[self.special_tokens['bos_token']]]

            for token in tokens:
                token_id = self.str_to_int.get(token, self.str_to_int[self.special_tokens['unk_token']])
                one_sequence.append(token_id)

            one_sequence.append(self.str_to_int[self.special_tokens['eos_token']])

            if truncation and self.model_max_length:
                one_sequence = one_sequence[:self.model_max_length]

            max_seq = max(max_seq, len(one_sequence)) #might not be used
            encodings.append(one_sequence)

        if padding:
            pad_id = self.str_to_int[self.special_tokens['pad_token']]
            encodings = [seq + [pad_id] * (max_seq - len(seq)) for seq in encodings]
            

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an 
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.
        masks = [[1 if t != self.str_to_int[self.special_tokens['pad_token']] else 0 for t in seq] for seq in encodings]

        if return_tensors == 'pt':
            return BatchEncoding({'input_ids': torch.tensor(encodings), 
                             'attention_mask': torch.tensor(masks)})
    
        return BatchEncoding({'input_ids': encodings, 
                             'attention_mask': masks})
        
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
    def __init__(self, vocab_size=None, embedding_size=None, hidden_size=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = torch.nn.Embedding(num_embeddings=config.vocab_size,
                                            embedding_dim=config.embedding_size)
        self.rnn = torch.nn.LSTM(input_size=config.embedding_size,
                                 hidden_size=config.hidden_size, batch_first=True) 
        #Batch first for (B=Batch size,N=sequence length,E=Embedding dimensionality)
        self.unembedding = torch.nn.Linear(in_features=config.hidden_size, 
                                           out_features=config.vocab_size)
        
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        train_loader = DataLoader(self.train_dataset, 
                                  batch_size=args.per_device_train_batch_size,
                                  shuffle=True)
        val_loader = DataLoader(self.eval_dataset, 
                                batch_size=args.per_device_eval_batch_size,
                                shuffle=True)
        
        # TODO: Your work here is to implement the training loop.
        self.model.train()
        for epoch in range(args.num_train_epochs):
            step = 0
            for batch in train_loader:
                #       PREPROCESSING AND FORWARD PASS:
                encodings = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
                input_ids = encodings['input_ids']

                X = input_ids[:, :-1]
                Y = input_ids[:, 1:]

                #       put X and Y onto the GPU (or whatever device you use)
                X = X.to(device)
                Y = Y.to(device)

                #       apply the model to X
                logit_results = self.model(X)

                #       compute the loss for the model output and Y
                loss = loss_func(logit_results.reshape(-1, logit_results.size(-1)), Y.reshape(-1))

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
                    enc = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
                    input_ids = enc['input_ids'].to(device)
                    attn = enc['attention_mask'].to(device)

                    X = input_ids[:, :-1]
                    Y = input_ids[:, 1:]
                    valid = attn[:, 1:]  # mask for Y to exclude padding tokens

                    logits = self.model(X)
                    loss = loss_func(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))

                    num_valid = valid.sum().item() #since 1 is non-padding token, summing here gives us all non-padding tokens
                    total_loss += loss.item() * num_valid
                    total_tokens += num_valid

                perplexity = float(np.exp(total_loss / total_tokens))
                print(f"#\nPerplexity for the epoch is: {perplexity:.3f}\n#")
                self.model.train()