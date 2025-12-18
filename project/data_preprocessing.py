from datasets import load_dataset, concatenate_datasets

def mapping(sample):
    tmp = sample["translation"]
    return_dict = {}

    for key, value in tmp.items():
        if key != "en":
            return_dict["non_english"] = value
        else:
            return_dict["english"] = value

    return return_dict


def _load_data(target_language="sv", nr_rows=None):
    url = f"https://huggingface.co/datasets/Helsinki-NLP/europarl/resolve/main/en-{target_language}"
    data_files_p1 = {"train": f"{url}/train-00000-of-00002.parquet"}
    dataset_p1 = load_dataset("parquet", data_files=data_files_p1, split="train")

    data_files_p2 = {"train": f"{url}/train-00001-of-00002.parquet"}
    dataset_p2 = load_dataset("parquet", data_files=data_files_p2, split="train")

    dataset = concatenate_datasets([dataset_p1, dataset_p2])
    dataset = dataset.map(mapping, remove_columns=["translation"])
    
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