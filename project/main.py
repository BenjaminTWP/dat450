from data_preprocessing import get_dataset, get_training_corpus_generator
from argparse import ArgumentParser
from tokenizer import train_trilingual_tokenizer, encode_dataset
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast, TrainingArguments



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
    parser.add_argument("--vocab-size", default=500000)
    parser.add_argument("--model-max-length", default=1028)
    parser.add_argument("--token-ds-out-path", default="tokenized_datasets/")

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
        first_dataset["train"] = encode_dataset(first_dataset["train"], tokenizer)
        first_dataset["test"] = encode_dataset(first_dataset["test"], tokenizer)

        second_dataset["train"] = encode_dataset(second_dataset["train"], tokenizer)
        second_dataset["test"] = encode_dataset(second_dataset["test"], tokenizer)

        print(f"\nSaving the tokenized data under the folder {args.token_ds_out_path}")
        first_dataset.save_to_disk(args.token_ds_out_path + "first_dataset_tokenized")
        second_dataset.save_to_disk(args.token_ds_out_path + "second_dataset_tokenized")


        
    elif args.run == "train":
        print("Loading tokenized datasets")
        first_dataset = load_from_disk(args.token_ds_out_path + "first_dataset_tokenized")
        second_dataset = load_from_disk(args.token_ds_out_path + "second_dataset_tokenized")
        print("Finished Loading tokenized datasets")
        
        print(first_dataset)

        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.token_output_dir)

        print(tokenizer.decode(first_dataset["train"][0]["input_ids_en"]))
        print(tokenizer.decode(first_dataset["train"][0]["input_ids_non_en"]))

        #TODO: Add code for training the model

    else: 
        raise Exception("The method you tried to call is not implemented")
