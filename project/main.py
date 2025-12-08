from data_preprocessing import load_data, train_test_data_split, get_training_corpus_generator
from argparse import ArgumentParser
from tokenizer import train_trilingual_tokenizer
from transformers import AutoTokenizer


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

    args = parser.parse_args()

    first_dataset = load_data(args.l1, nr_rows=args.data_limit)
    first_dataset_train_test_split = train_test_data_split(first_dataset, args.split_size)

    second_dataset = load_data(args.l1, nr_rows=args.data_limit)
    second_dataset_train_test_split = train_test_data_split(second_dataset, split_size=args.split_size)

    if args.run == "tokenizer":
        print("Creating tokenizer")
        generator = get_training_corpus_generator(
            first_dataset_train_test_split["train"],
            second_dataset_train_test_split["train"]
        )

        train_trilingual_tokenizer(
            generator,
            args.token_output_dir, 
            args.model_max_length, 
            args.vocab_size
        )

    elif args.run == "test tokenizer":
        tokenizer = AutoTokenizer.from_pretrained(args.token_output_dir)

        examples = [
            "Hej jag heter Bertil. Kan du säga mig vem som är tomten?",
            "I have a fat cat",
            "Che una birra grande, mi piace"
        ]

        for example in examples:
            encoding = tokenizer(example)
            print(tokenizer.decode(encoding['input_ids']))

