from datasets import load_dataset
from ..a1.A1_skeleton import build_tokenizer
import nltk


def load_data(target_language="sv"):
    url = f"https://huggingface.co/datasets/sentence-transformers/parallel-sentences-jw300/resolve/main/en-{target_language}"
    data_files = {"train": f"{url}/train-00000-of-00001.parquet"}
    return load_dataset("parquet", data_files=data_files, split="train")

def train_test_data_split(data, split_size=0.2, seed=42):
    shuffled_data = data.shuffle(seed)
    return shuffled_data.train_test_split(split_size)




en_sv_data = load_data("sv")
en_sv_train_test_data = train_test_data_split(en_sv_data)

en_it_data = load_data("it")
en_it_train_test_data = train_test_data_split(en_it_data)


