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

from model import LanguageTransformer, ModelConfig, translate_sentence
import torch


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

    args = parser.parse_args()
    
 
    if args.run == "tokenizer":
        
        first_dataset = get_dataset(args.l1, args.data_limit, args.split_size)
        second_dataset = get_dataset(args.l2, args.data_limit, args.split_size)

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

        training_args = TrainingArguments(lr=0.001, epochs=3, batch_size=32)

        project_trainer = ProjectTrainer(model=model, args=training_args, dataset=first_dataset, tokenizer=tokenizer)

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

        model = LanguageTransformer.from_pretrained("trained_model").to(device)

        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.token_output_dir)

        translation = translate_sentence(model, source_sentence, tokenizer, device, max_length=50)

        print(translation)


    else: 
        raise Exception("The method you tried to call is not implemented")
