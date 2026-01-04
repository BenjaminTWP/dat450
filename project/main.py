from data_preprocessing import get_dataset, get_training_corpus_generator
from tokenizer import train_trilingual_tokenizer, encode_dataset
from datasets import load_from_disk
from trainer import TrainingArguments, ProjectTrainer
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import evaluate
from torch.utils.data import DataLoader
from model import LanguageTransformer, ModelConfig
from translate import translate_sentence, translate_tokens_batch
import torch
from tqdm import tqdm
from runtime_args import get_args


if __name__ == "__main__":
    
    args = get_args()
 
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

    elif args.run == "encode dataset": 

        if args.is_helsinki:
            print(f"Tokenize dataset using the following model from Helsinki research group {args.helsinki_model}")
            tokenizer = AutoTokenizer.from_pretrained(args.helsinki_model)
        else: 
            print("Tokenize dataset using custom model")
            tokenizer = PreTrainedTokenizerFast.from_pretrained(args.token_output_dir)

        first_dataset = get_dataset(args.l1, args.data_limit, args.split_size)
        second_dataset = get_dataset(args.l2, args.data_limit, args.split_size)

        print("\nStarting data tokenization")
        first_dataset["train"] = encode_dataset(first_dataset["train"], tokenizer, args.batch_size)
        first_dataset["test"] = encode_dataset(first_dataset["test"], tokenizer, args.batch_size)

        second_dataset["train"] = encode_dataset(second_dataset["train"], tokenizer, args.batch_size)
        second_dataset["test"] = encode_dataset(second_dataset["test"], tokenizer, args.batch_size)

        print(f"\nSaving the tokenized data under the folder {args.token_ds_out_path}")
        first_dataset.save_to_disk(args.token_ds_out_path + f"{args.l1}_en_dataset_tokenized")
        second_dataset.save_to_disk(args.token_ds_out_path + f"{args.l2}_en_dataset_tokenized")

        
    elif args.run == "train":
        print("Loading tokenized datasets")
        dataset = load_from_disk(args.token_ds_out_path + args.dataset_load_name)
        print("Finished Loading tokenized datasets")

        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.token_output_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if args.load_model_dir:
            model = LanguageTransformer.from_pretrained(args.load_model_dir)
            print(f"Model loaded from {args.load_model_dir}")
        
        else:   
            config = ModelConfig(
                vocab_size=args.vocab_size, 
                hidden_size=args.hidden_size, 
                intermediate_size=args.intermediate_size, 
                num_attention_heads=args.num_attention_heads, 
                num_hidden_layers=args.num_hidden_layers,
                rope_theta=2, 
                hidden_act='silu', 
                max_position_embeddings=1000, 
                rms_norm_eps=args.rms_norm_eps
            )
            
            model = LanguageTransformer(config)
            print("Initialized new model")

        training_args = TrainingArguments(
            lr=args.lr,
            epochs=args.epochs, 
            batch_size=args.batch_size
        )

        project_trainer = ProjectTrainer(
            model=model, 
            args=training_args, 
            dataset=dataset, 
            tokenizer=tokenizer, 
            output_dir=args.save_model_dir
        )

        project_trainer.train()


    elif args.run == "params":

        config = ModelConfig(
            vocab_size=args.vocab_size, 
            hidden_size=args.hidden_size, 
            intermediate_size=args.intermediate_size, 
            num_attention_heads=args.num_attention_heads, 
            num_hidden_layers=args.num_hidden_layers,
            rope_theta=2, 
            hidden_act='silu', 
            max_position_embeddings=1000, 
            rms_norm_eps=args.rms_norm_eps
        )
        
        model = LanguageTransformer(config)
        total_params = sum(p.numel() for p in model.parameters())
        print("Total number of parameters in the model: ", total_params)
        
    
    elif args.run == "gen":

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if args.is_helsinki:
            print(f"Loading the following model from Helsinki research group {args.helsinki_model}")
            tokenizer = AutoTokenizer.from_pretrained(args.helsinki_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.helsinki_model)
        else: 
            print("Loading custom model")
            tokenizer = PreTrainedTokenizerFast.from_pretrained(args.token_output_dir)
            model = LanguageTransformer.from_pretrained(args.load_model_dir)
        
        source_sentence = input("Welcome to ChatGBG, what do you want to translate? \n - ")

        translation = translate_sentence(
            model.to(device), 
            source_sentence, 
            tokenizer, 
            device, 
            args.gen_len,
            args.is_helsinki
        )

        print(translation)


    elif args.run == "eval":

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading tokenized datasets")
        dataset = load_from_disk(args.token_ds_out_path + args.dataset_load_name)

        eval_dataset = dataset["test"]

        if args.is_helsinki:
            print(f"Loading the following model from Helsinki research group {args.helsinki_model}")
            tokenizer = AutoTokenizer.from_pretrained(args.helsinki_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.helsinki_model)
        else: 
            print("Loading custom model")
            tokenizer = PreTrainedTokenizerFast.from_pretrained(args.token_output_dir)
            model = LanguageTransformer.from_pretrained(args.load_model_dir)

        model = model.to(device)

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        val_loader = DataLoader(eval_dataset, 
                                batch_size=50,
                                shuffle=False,
                                collate_fn=data_collator)

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

            gen_eng = translate_tokens_batch(
                model, 
                source_lang_tokens, 
                attention_mask, 
                tokenizer,
                device,
                args.gen_len,
                args.is_helsinki
            )

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
        
        print("Computing comet score. This may take a while (~20 min using L40s)")
        comet_result = comet.compute()
        print("Comet: ", comet_result["mean_score"])


    elif args.run == "translate examples":
        
        # Note: The translation of these sentences will be incorrect for one language if
        # the model has not been trained on both languages. Make sure to load such a model

        text_to_translate = [
            "Vi har ett möte klockan fem om de ökande kostnaderna for energiproduktionen.",
            "Abbiamo una riunione alle cinque sull'aumento dei costi della produzione energetica.",
            "Det kommande mötet kommer äga rum utanför Paris.",
            "Il prossimo incontro si terrà fuori Parigi.", 
            "Den punkt du nämnde är mycket relevant för frågan.",
            "Il punto che hai menzionato è molto rilevante per la domanda"
        ]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LanguageTransformer.from_pretrained(args.load_model_dir).to(device)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.token_output_dir)
        
        for i in range(0, len(text_to_translate), 2):

            translation_sv = translate_sentence(model, text_to_translate[i], tokenizer, device, max_length=200)
            translation_it = translate_sentence(model, text_to_translate[i+1], tokenizer, device, max_length=200)

            print(f"Swedish translation: {translation_sv}")
            print(f"Italian translation: {translation_it}")
            print("-" * 200)


    else: 
        raise Exception("The method you tried to call is not implemented")
