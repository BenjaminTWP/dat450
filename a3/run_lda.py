import os
import nltk
import numpy as np
import torch
from tqdm import tqdm
import string
import nltk

nltk.download('reuters')

from nltk.corpus import stopwords
from collections import Counter

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]


def get_stopwords():
    stop = set(stopwords.words("english"))
    stop = stop.union(set(string.punctuation))
    stop = stop.union({'``', "''", "'s", "dlrs", "pct", "cts", 'lt', 'mln', 'said'})
    return stop


def create_data_set(dir="training", min_freq=10, corpus_limit=200000, number_of_topics = 15):

    path = os.path.join(os.getcwd(), "reuters", dir)
    files = os.listdir(path)
    files = sorted(files, key=lambda x: int(x))

    docs = []
    word_counter = Counter()
    total_words = 0

    for file in files:

        
        file = os.path.join(path, file)
        with open(file, 'r') as f:
            raw_file = f.readlines()

            file_words = []

            for raw_line in raw_file:
                new_words = lowercase_tokenizer(raw_line)
                file_words.extend(new_words)

            docs.append(file_words)
            word_counter.update(file_words)
            total_words += len(file_words)

        if total_words > corpus_limit:
            break

    uncommon_words = [item for item, count in word_counter.items() if count <= min_freq]

   
    stop_words =  get_stopwords()

    words_to_remove = stop_words.union(set(uncommon_words))

    topic_map = {}

    for i, doc in enumerate(docs):
        new_doc = []
        j = 0
        for word in doc:
            if word not in words_to_remove:
                new_doc.append(word)    
                topic_map[f"{i},{j}"] = np.random.randint(0, number_of_topics)
                j += 1
        docs[i] = new_doc

    return docs, topic_map, total_words


def create_word_mappings(docs):
    words = set()
    for doc in docs:
        words.update(doc)
    str_to_int = {}
    int_to_str = {}
    for i, word in enumerate(words):
        str_to_int[word] = i
        int_to_str[i] = word
    return str_to_int, int_to_str

def get_n_d_k(docs, topic_map, k):
    n_d_k = np.zeros((len(docs),k))
    for i, doc in enumerate(docs):
        for j in range(len(doc)):
            topic = topic_map[f"{i},{j}"]
            n_d_k[i, topic] += 1
    return n_d_k

def get_m_k_v(docs, topic_map, k, str_to_int):
    m_k_v = np.zeros((k, len(str_to_int)))
    for i, doc in enumerate(docs):
        for j, word in enumerate(doc):
            topic = topic_map[f"{i},{j}"]
            m_k_v[topic, str_to_int[word]] += 1
            
    return m_k_v

def n_dj_k(topic_map, n_d_k, d, j, k):
    
    topic = topic_map[f"{d},{j}"]

    update = 0
    if topic == k:
        update = -1

    count = n_d_k[d, k]
    return  count + update

def m_dj_w(topic_map, m_k_v, d, j, k, w_index):
    
    topic = topic_map[f"{d},{j}"]

    update = 0
    if topic == k:
        update = -1

    count = m_k_v[k, w_index]

    return  count + update

def m_dj(topic_map, m_k, d, j, k):
    
    topic = topic_map[f"{d},{j}"]

    update = 0
    if topic == k:
        update = -1

    count = m_k[k]

    return  count + update

def occurance_counts(docs):
    doc_sets = [set(doc) for doc in docs]

    cnt = Counter()
    for doc in doc_sets:
        cnt.update(doc)


    co_occurance_count = Counter()
    for doc_words in doc_sets:
        for w1 in doc_words:
            for w2 in doc_words:
                if w1 != w2:
                    co_occurance_count[f"{w1},{w2}"] += 1
    
    return dict(co_occurance_count), dict(cnt)


def umass(co_occurance, occurance, common_words_topic):
    score = 0
    for m in range(1, len(common_words_topic)):
        w1, _ = common_words_topic[m]
        for l in range(0, m-1):
            w2, _ = common_words_topic[l]
            co_occur = co_occurance.get(f"{w1},{w2}", 0) + 1
            d_count = occurance.get(f"{w2}", 0)
            score += np.log(np.divide(co_occur, d_count))
            
    return score


def get_related_document_words(docs, topic_map):

    related_words = []
    for i, doc in enumerate(docs):
        topic_splits = {}
        for j, word in enumerate(doc):
            topic = topic_map[f"{i},{j}"]
            topic_splits.setdefault(topic, []).append(word)
        related_words.append(topic_splits)

    return related_words


def get_top_words_per_topic(related_words, n_top_words=20):
    combined_doc_topics = {}
    for doc in related_words:
        for key, array in doc.items():
            combined_doc_topics.setdefault(key, []).extend(array)

    
    most_common_words = {}
    for topic, words in combined_doc_topics.items():
        cnt = Counter()
        cnt.update(words)
        most_common_words[topic] = [(token, count) for token, count in cnt.most_common(n_top_words)]

    return most_common_words



def train(nr_iterations, nr_words, docs, str_to_int, 
        k, topic_map, alpha, beta, m_k, n_d_k, m_k_v,
        co_occurance, occurance
    ):
    print("Starting iterations", flush=True)
    d = len(docs)
    scores = np.zeros((nr_iterations, k))
    for iteration in range(nr_iterations):
        for _ in range(nr_words):
            r_d = 0    
            doc_length = 0
            while doc_length == 0:
                r_d = np.random.randint(0, d)
                doc_length = len(docs[r_d])

            r_j = np.random.randint(0, len(docs[r_d]))

            q = np.zeros(k)
            p = np.zeros(k)
            w_index = str_to_int[docs[r_d][r_j]]
            vocab_len = len(str_to_int)

            for k_i in range(k):
                temp_n_dj_k = n_dj_k(topic_map, n_d_k, r_d, r_j, k_i)
                temp_m_dj_w = m_dj_w(topic_map, m_k_v, r_d, r_j, k_i, w_index)
                temp_m_dj = m_dj(topic_map, m_k, r_d, r_j, k_i)
                
                q[k_i] = (alpha+temp_n_dj_k)*(beta+temp_m_dj_w) / (vocab_len*beta + temp_m_dj)

            q_sum = q.sum()
            for k_i in range(k):
                p[k_i] = q[k_i] / q_sum

            
            dist = torch.distributions.Categorical(torch.tensor(p))
            new_z = dist.sample().item()

            old_topic = topic_map[f"{r_d},{r_j}"]
            topic_map[f"{r_d},{r_j}"] = new_z

            n_d_k[r_d, old_topic] -= 1
            m_k_v[old_topic, w_index] -= 1

            n_d_k[r_d, new_z] += 1
            m_k_v[new_z, w_index] += 1

            m_k = m_k_v.sum(axis=1)

        related_words = get_related_document_words(docs, topic_map)
        common_words_topic = get_top_words_per_topic(related_words)
        for k_i in range(k):
            scores[iteration,k_i] = umass(co_occurance, occurance, common_words_topic[k_i])    

        print(f"For iteration {iteration} the mean umass score was: {scores[iteration].mean()}", flush=True)

    return topic_map, scores


def occurance_counts(docs):
    doc_sets = [set(doc) for doc in docs]


    unique_words = set()
    for doc_set in doc_sets:
        unique_words.update(doc_set)

    cnt = Counter()
    for doc in doc_sets:
        cnt.update(doc)


    co_occurance_count = Counter()
    for doc_words in doc_sets:
        for w1 in doc_words:
            for w2 in doc_words:
                if w1 != w2:
                    co_occurance_count[f"{w1},{w2}"] += 1
    
    return dict(co_occurance_count), dict(cnt), unique_words