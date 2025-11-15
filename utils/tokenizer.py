import torch, nltk, pickle
nltk.download('punkt')
nltk.download('punkt_tab')
from collections import Counter
from transformers import BatchEncoding

###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]


def create_vocabulary(tokenized_words, max_voc_size, pad_token, unk_token, bos_token, eos_token):
    cnt = Counter()
    cnt.update(tokenized_words)

    #  More words than allowed then keep only the most frequently used ones
    max_voc_size = len(cnt) + 5 if not max_voc_size else max_voc_size
    if len(cnt) > max_voc_size-4:
        cnt = Counter(cnt.most_common(max_voc_size-4))

    cnt_list = list(cnt)
    print("Most common: ", cnt_list[:5])
    print("Least common: ", cnt_list[-5:])
    
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
            tokenized_words = lowercase_tokenizer(text)
            tokenized_words.insert(0, self.bos_token)
            tokenized_words.append(self.eos_token)
            
            if truncation and self.model_max_length and text_length > self.model_max_length:
                text = text[:self.model_max_length]
            
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
   