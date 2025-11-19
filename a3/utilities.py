from nltk.corpus import stopwords
from collections import Counter
import os
import nltk
import string

nltk.download('stopwords')
reuter_stop_words = "/data/users/benpe/downloads/data/reuters/stopwords"

def get_stop_words():
    sw = set(stopwords.words('english'))
    sw |= set(string.punctuation)
    sw |= {'``', "''", "'s", "dlrs", "pct", "cts", 'lt', 'mln'}

    with open(reuter_stop_words, encoding="utf-8") as f:
        for line in f:
            sw.add(line.strip().lower())

    return sw

def lowercase_tokenizer(text, stopwords):
    tokens = [t.lower() for t in nltk.word_tokenize(text)]
    return [t for t in tokens if t not in stopwords]


def integer_encoding(counter):
    str_to_int = {}
    int_to_str = {}
    for i, key in enumerate(counter):
        str_to_int.update({key: i})
        int_to_str.update({i: key})
    return str_to_int, int_to_str


def get_all_occurences(train_loc, n_min=10):
    stopwords = get_stop_words()
    word_counts = Counter()

    for filename in os.listdir(train_loc):
        file_path = os.path.join(train_loc, filename)  
        with open(file_path, encoding="utf-8") as file:

            words = []
            for line in file:
                words += lowercase_tokenizer(line.strip(), stopwords)
            word_counts.update(words)

    return Counter({word:count for word, count in word_counts.items() if count >= n_min})


def tokenize_document(train_loc, document):
    stopwords = get_stop_words()
    word_counts = Counter()

    file_path = os.path.join(train_loc, document)
    with open(file_path, encoding="utf-8") as file:
        words = []
        for line in file:
            words += lowercase_tokenizer(line.strip(), stopwords)
        word_counts.update(words)
    
    return word_counts

def encode_docs(train_loc, str_to_int):
    stopwords = get_stop_words()
    docs = []
    for filename in os.listdir(train_loc):
        file_path = os.path.join(train_loc, filename)
        with open(file_path, encoding="utf-8") as file:
            words = []
            for line in file:
                words += lowercase_tokenizer(line.strip(), stopwords)
        docs.append([str_to_int[word] for word in words if word in str_to_int])
    return docs