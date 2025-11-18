from utilities import *
from lda import lda
import numpy as np

train_loc = "/data/users/benpe/downloads/data/reuters/training"

if __name__ == "__main__": 

    ###¤## SETUP
    counter_all = get_all_occurences(train_loc, n_min=2)
    str_to_int, int_to_str = integer_encoding(counter_all)

    docs = encode_docs(train_loc, str_to_int)
    vocab_size = len(counter_all)
    n_documents = len(docs)
    k_topics = 20
    alpha=0.01
    beta=0.01
