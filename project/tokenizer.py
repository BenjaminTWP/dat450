from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import os

from tokenizers import(
    normalizers,
    processors
)

BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"

def train_trilingual_tokenizer(data_generator, save_dir, model_max_length, vocab_size):
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

    tokenizer.normalizer = normalizers.NFD() 

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=True, 
        trim_offsets=True,
        use_regex=True
    )

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN],
    )
    
    tokenizer.train_from_iterator(data_generator, trainer=trainer)    

    bos_token_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN}:0 $A:0 {EOS_TOKEN}:0",
        special_tokens=[(BOS_TOKEN, bos_token_id), (EOS_TOKEN, eos_token_id)]
    )

    tokenizer.decoder = ByteLevelDecoder()

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN
    )

    
    hf_tokenizer.model_max_length = model_max_length
    hf_tokenizer.save_pretrained(save_dir)


def encode_dataset(
        dataset, 
        tokenizer,
        batch_size, 
        padding="longest", 
        truncation="longest_first",
        max_length=256, 
        return_tensor="pt"
    ):

    def encode(batch):
        english = tokenizer(
            batch["english"],
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensor,
        ), 

        non_english = tokenizer(
            batch["non_english"],
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensor,
        ), 

        english = english[0]
        non_english = non_english[0]

        return {
            "input_ids_en": english["input_ids"],
            "attention_mask_en": english["attention_mask"],
            "input_ids_non_en": non_english["input_ids"],
            "attention_mask_non_en": non_english["attention_mask"],
        }

    print(os.cpu_count())
    tokenized_dataset = dataset.map(
        encode,
        batched=True,
        batch_size=batch_size,
        remove_columns=["english", "non_english"],
        num_proc=os.cpu_count()
    )

    return tokenized_dataset
    