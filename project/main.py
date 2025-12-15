from data_preprocessing import get_dataset, get_training_corpus_generator
from argparse import ArgumentParser
from tokenizer import train_trilingual_tokenizer, encode_dataset
from datasets import load_from_disk
from transformers import (
    PreTrainedTokenizerFast, 
    GenerationConfig, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from model import LanguageTransformerForCausalLM
from model import ModelConfig
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
    parser.add_argument("--vocab-size", default=500000)
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
        device = "cuda" if torch.cuda.is_available() else "cpu",
        device = device[0]

        config = ModelConfig(
                vocab_size=args.vocab_size, 
                hidden_size=128, 
                intermediate_size=256, 
                num_attention_heads=4, 
                num_hidden_layers=5,
                rope_theta=2, 
                hidden_act='silu', 
                max_position_embeddings=1000, 
                rms_norm_eps=0.001)
        
        model = LanguageTransformerForCausalLM(config).to(device)


        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        #batch = data_collator([first_dataset["train"][i] for i in range(1, 3)])
        batch = data_collator([first_dataset["train"][0]])

        # Show the prompt in the source language
        print("\nThe following is the source language")
        print(tokenizer.decode(batch["input_ids"].squeeze(0)))

        # Show the prompt in the target language
        print("\nThe following is the target language")
        print(tokenizer.decode(batch["labels"].squeeze(0)))


        generation_config = GenerationConfig(
            max_new_tokens=30
        )

        training_config = Seq2SeqTrainingArguments(
            output_dir="model_checkpoints",
            eval_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=0.01,
            num_train_epochs=1,
            logging_strategy="steps",
            sortish_sampler=True,
            predict_with_generate=True,
            dataloader_num_workers=4,
            generation_config=generation_config,
            save_strategy="best",
            metric_for_best_model="loss",
            greater_is_better=False,
            neftune_noise_alpha=0.1
        )

        trainer = Seq2SeqTrainer(
            model=model, 
            args=training_config,
            data_collator=data_collator,
            train_dataset=first_dataset["train"],
            eval_dataset=first_dataset["test"]
        )


        en_tokens = torch.tensor(first_dataset["train"][0]["input_ids"]).unsqueeze(0).to(device)
        non_en_tokens = torch.tensor(first_dataset["train"][0]["labels"]).unsqueeze(0).to(device)
        bos_decoder_tensor = torch.tensor(tokenizer.encode("Detta är ett test för att kolla om...")).unsqueeze(0).to(device)
        
        # TODO: The generate method does not generate for the target language, rather it generates 
        # for the source language. Has to be updated before we train the model

        print("\nGenerating output")
        output_ids = model.generate(
            input_ids=en_tokens,
            decoder_input_ids=non_en_tokens[:, :5],
            generation_config=generation_config
        )
        
        print(tokenizer.decode(output_ids.squeeze(0)))

    else: 
        raise Exception("The method you tried to call is not implemented")
