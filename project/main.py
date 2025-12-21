from data_preprocessing import get_dataset, get_training_corpus_generator
from argparse import ArgumentParser
from tokenizer import train_trilingual_tokenizer, encode_dataset
from datasets import load_from_disk
from trainer import TrainingArguments, ProjectTrainer
from transformers import (
    PreTrainedTokenizerFast, 
    GenerationConfig, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate
from torch.utils.data import DataLoader
from model import LanguageTransformer, ModelConfig
from translate import translate_sentence, translate_tokens, translate_tokens_batch
import torch
from tqdm import tqdm


if __name__ == "__main__":
    parser = ArgumentParser()


    parser.add_argument("--run", default="tokenizer")
    parser.add_argument("--token-output-dir", default="hf_compatible_trilingual_tokenizer")

    # Note that we assume that english is the language we translate from so these are just
    # the languages we are translating to, i.e we have two datasets en -> sv and en -> it
    parser.add_argument("--l1", help="the first language we want to use", default="sv")
    parser.add_argument("--l2", help="the second language we want to use", default="it")

    parser.add_argument("--data-limit", default=None)
    parser.add_argument("--split-size", help="Percentage of test data", default=0.2)
    parser.add_argument("--vocab-size", default=50000)
    parser.add_argument("--model-max-length", default=1028)
    parser.add_argument("--token-ds-out-path", default="tokenized_datasets/")
    parser.add_argument("--batch-size", default=32)
    parser.add_argument("--load-model-dir", default=None)

    args = parser.parse_args()
    
 
    if args.run == "tokenizer":
        
        first_dataset = get_dataset(args.l1, args.data_limit, args.split_size) # Swedish
        second_dataset = get_dataset(args.l2, args.data_limit, args.split_size) # Italian

        print("\nCreating tokenizer")
        generator = get_training_corpus_generator(
            first_dataset["train"],
            second_dataset["train"]
        )

        train_trilingual_tokenizer(
            generator,
            args.token_output_dir, 
            args.model_max_length, 
            args.vocab_size
        )

    elif args.run == "test tokenizer":
        print("Testing tokenizer")

        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.token_output_dir)

        examples = [
            "Hej jag heter Bertil. Kan du säga mig vem som är tomten?",
            "I have a fat cat",
            "Che una birra grande, mi piace"
        ]

        for example in examples:
            encoding = tokenizer(example)
            decoded_text = tokenizer.decode(encoding["input_ids"])
            print(decoded_text)

    elif args.run == "encode dataset": 
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.token_output_dir)

        first_dataset = get_dataset(args.l1, args.data_limit, args.split_size)
        second_dataset = get_dataset(args.l2, args.data_limit, args.split_size)

        print("\nStarting data tokenization")
        first_dataset["train"] = encode_dataset(first_dataset["train"], tokenizer, args.batch_size)
        first_dataset["test"] = encode_dataset(first_dataset["test"], tokenizer, args.batch_size)

        second_dataset["train"] = encode_dataset(second_dataset["train"], tokenizer, args.batch_size)
        second_dataset["test"] = encode_dataset(second_dataset["test"], tokenizer, args.batch_size)

        print(f"\nSaving the tokenized data under the folder {args.token_ds_out_path}")
        first_dataset.save_to_disk(args.token_ds_out_path + "first_dataset_tokenized")
        second_dataset.save_to_disk(args.token_ds_out_path + "second_dataset_tokenized")


        
    elif args.run == "train":
        print("Loading tokenized datasets")
        first_dataset = load_from_disk(args.token_ds_out_path + "first_dataset_tokenized")
        second_dataset = load_from_disk(args.token_ds_out_path + "second_dataset_tokenized")
        print("Finished Loading tokenized datasets")


        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.token_output_dir)

        #TODO: Add code for training the model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if args.load_model_dir:
            model = LanguageTransformer.from_pretrained(args.load_model_dir)
            print(f"Model loaded from {args.load_model_dir}")
        
        else:   
            config = ModelConfig(
                    vocab_size=args.vocab_size, 
                    hidden_size=256, 
                    intermediate_size=512, 
                    num_attention_heads=4, 
                    num_hidden_layers=5,
                    rope_theta=2, 
                    hidden_act='silu', 
                    max_position_embeddings=1000, 
                    rms_norm_eps=0.001)
            
            model = LanguageTransformer(config)
            print("Initialized new model")

        training_args = TrainingArguments(lr=0.0001, epochs=5, batch_size=32)

        project_trainer = ProjectTrainer(model=model, args=training_args, dataset=second_dataset, tokenizer=tokenizer, output_dir="trained_model_e5_it_masking")

        project_trainer.train()


    elif args.run == "params":

        config = ModelConfig(
                vocab_size=args.vocab_size, 
                hidden_size=256, 
                intermediate_size=512, 
                num_attention_heads=4, 
                num_hidden_layers=5,
                rope_theta=2, 
                hidden_act='silu', 
                max_position_embeddings=1000, 
                rms_norm_eps=0.001)
        
        model = LanguageTransformer(config)
        total_params = sum(p.numel() for p in model.parameters())
        print("Total number of parameters in the model: ", total_params)


    
    elif args.run == "gen":
        source_sentence = input("Welcome to ChatGBG, what do you want to translate? \n - ")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = LanguageTransformer.from_pretrained("trained_model_e5_sv_masking").to(device)

        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.token_output_dir)

        translation = translate_sentence(model, source_sentence, tokenizer, device, max_length=50)

        print(translation)


    elif args.run == "eval_b":

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading tokenized datasets")
        first_dataset = load_from_disk(args.token_ds_out_path + "first_dataset_tokenized")
        second_dataset = load_from_disk(args.token_ds_out_path + "second_dataset_tokenized")
    
        eval_dataset = first_dataset["test"]

        model = LanguageTransformer.from_pretrained("trained_model_e5_sv_masking").to(device)

        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.token_output_dir)

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        val_loader = DataLoader(eval_dataset, 
                                batch_size=50,
                                shuffle=False,
                                collate_fn=data_collator)

        print("Finished Loading tokenized datasets")

        sacrebleu = evaluate.load("sacrebleu")
        chrf = evaluate.load("chrf")
        comet = evaluate.load('comet', model_name='wmt20-comet-da')

        count = 0
        for batch in tqdm(val_loader):
            
            source_lang_tokens = batch["input_ids"].to(device)
            true_tokens = batch["labels"]
            attention_mask = batch["attention_mask"].to(device)

            true_tokens = true_tokens.masked_fill(true_tokens == -100, tokenizer.pad_token_id).to(device)

            source_sentences =  tokenizer.batch_decode(source_lang_tokens, skip_special_tokens=True)
            true_eng = tokenizer.batch_decode(true_tokens, skip_special_tokens=True)

            gen_eng = translate_tokens_batch(model, source_lang_tokens, attention_mask, tokenizer, device, max_length=200)

            ref_format = [[sentence] for sentence in true_eng]

            sacrebleu.add_batch(
                predictions=gen_eng,
                references=ref_format
            )

            chrf.add_batch(
                predictions=gen_eng,
                references=ref_format
            )

            comet.add_batch(
                sources=source_sentences,
                predictions=gen_eng,
                references=true_eng,
            )

            del source_lang_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sacrebleu_result = sacrebleu.compute()
        chrf_result = chrf.compute()

        print("sacrebleu: ", sacrebleu_result["score"])
        print("CHRF: ", chrf_result["score"])
        
        print("Computing comet score. This may take a while")
        comet_result = comet.compute()
        print("Comet: ", comet_result["mean_score"])


    else: 
        raise Exception("The method you tried to call is not implemented")
