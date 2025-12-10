from datasets import load_dataset

def _load_data(target_language="sv", nr_rows=None):
    print(f"Loading dataset en -> {target_language}")
    url = f"https://huggingface.co/datasets/sentence-transformers/parallel-sentences-jw300/resolve/main/en-{target_language}"
    data_files = {"train": f"{url}/train-00000-of-00001.parquet"}
    dataset = load_dataset("parquet", data_files=data_files, split="train")
    if nr_rows:
        return dataset.select(range(int(nr_rows)))
    return dataset

def _train_test_data_split(data, split_size=0.2, seed=42):
    shuffled_data = data.shuffle(seed)
    return shuffled_data.train_test_split(split_size)


def get_training_corpus_generator(dataset_1, dataset_2, load_size=1000):
    d1_len = len(dataset_1)
    d2_len = len(dataset_2)    
    use_english_from_ds1 = True if d1_len > d2_len else False

    for index in range(0, max(d1_len, d2_len), load_size):
        upper_slice_idx = index + load_size

        if upper_slice_idx < d1_len:
            samples = dataset_1[index : upper_slice_idx]
            yield samples["non_english"]

            if use_english_from_ds1:
                yield samples["english"]

        if upper_slice_idx < d2_len:
            samples = dataset_2[index : upper_slice_idx]
            yield samples["non_english"]

            if not use_english_from_ds1:
                yield samples["english"]            


def get_dataset(language1, data_limit, split_size):
    dataset = _load_data(language1, nr_rows=data_limit)
    split_dataset = _train_test_data_split(dataset, split_size)
    
    print("-" * 80)
    print(f"Dataset for en -> {language1}")
    print(f"The total number of samples in the dataset is: {len(split_dataset["train"]) + len(split_dataset["test"])}\n")
    print(split_dataset)
    print("-" * 80)

    return split_dataset