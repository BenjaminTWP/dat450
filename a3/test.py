import matplotlib.pyplot as plt
from collections import Counter
from utilities import *
import numpy as np
from lda import *

train_loc = "/data/users/benpe/downloads/data/reuters/training"

def section(title):
    print(f"\n################################ {title} ###############################\n", flush=True)


def plot_coherence_over_iterations(coherence_history, k, alpha):
    iterations = [item[0] for item in coherence_history]
    mean_umass = [np.mean(item[1]) for item in coherence_history]

    plt.plot(iterations, mean_umass)
    plt.xlabel("Iterations")
    plt.ylabel("Mean umass score")
    plt.title(f"umass coherence over iterations. K_topics={k}. alpha=beta={alpha}")
    plt.savefig(f"mean_umass_k={k}_a=b={alpha}.png")

if __name__ == "__main__": 
    section("Tokenize, count all words, remove stopwords")
    counter_all = get_all_occurences(train_loc, n_min=5)
    str_to_int, int_to_str = integer_encoding(counter_all)
    print("Most common words: ", counter_all.most_common(10))
    print("Note that mln=million, dlrs=dollars, lt=long term, ct=cent, pct=percent. I.e. we are seeing abbreviations.")


    section("Encode all documents and set hyperparameters")
    docs = encode_docs(train_loc, str_to_int)
    vocab_size = len(counter_all)
    n_documents = len(docs)
    k_topics = 50
    alpha=0.1
    beta=alpha
    iterations = 180
    print(f"Size of vocabulary is {vocab_size}. Number of topics is {k_topics}. alpha=beta={alpha}. Running for {iterations} iterations.", flush=True)

    section("Initialize LDA variables, i.e. assign randomly and such")
    topic_assignments, document_topic_counts, topic_word_counts, topic_counts = init_lda(docs, k_topics, vocab_size)
    print("Example topic assignment for document 5:", topic_assignments[5])


    section("Computing word co-occurrence counts (mainly to calc umass)")
    cooccurrence_counts, word_document_counts = word_occurence_count(docs, vocab_size)
    print("Most common co-occurences in documents: ", [((int_to_str[cooc[0][0]], int_to_str[cooc[0][1]]), cooc[1]) for cooc in cooccurrence_counts.most_common(5)])
    

    section("collapsed Gibbs sampling") # note: coherence score is per topic per iteration. 
    coherence_history = gibbs_sampling(docs, topic_assignments, document_topic_counts, topic_word_counts,
                                                          topic_counts, alpha, beta, vocab_size, n_iterations=iterations, int_to_str=int_to_str,
                                                            cooccurrence_counts=cooccurrence_counts, word_document_counts=word_document_counts, top_n=20)

    
    section("Top words per topic:")
    phi = compute_phi(topic_word_counts, beta, vocab_size)
    _, last_umass = coherence_history[-1]
    for k in range(k_topics):
        top_ids = np.argsort(phi[k])[::-1][:20]
        top_words = [int_to_str[idx] for idx in top_ids]

        umass= last_umass[k]
        print(f"Topic {k+1} (last score={umass:.2f}): {top_words}")

    plot_coherence_over_iterations(coherence_history, k_topics, alpha)