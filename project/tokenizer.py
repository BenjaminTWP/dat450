from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from tokenizers import(
    pre_tokenizers,
    normalizers,
    processors
)


def train_trilingual_tokenizer(data_generator, save_dir, model_max_length, vocab_size):

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(), 
        normalizers.Lowercase(), 
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(), 
        pre_tokenizers.Punctuation()
    ])

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
    )

    tokenizer.train_from_iterator(data_generator, trainer=trainer)

    bos_token_id = tokenizer.token_to_id("<BOS>")
    eos_token_id = tokenizer.token_to_id("<EOS>")

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"<BOS>:0 $A:0 <EOS>:0",
        special_tokens=[("<BOS>", bos_token_id), ("<EOS>", eos_token_id)],
    )

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<BOS>",
        eos_token="<EOS>",
        unk_token="<UNK>",
        pad_token="<PAD>"
    )

    
    hf_tokenizer.model_max_length = model_max_length
    hf_tokenizer.save_pretrained(save_dir)