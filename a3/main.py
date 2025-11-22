from run_lda import (
    create_data_set,
    create_word_mappings,
    get_n_d_k,
    get_m_k_v,
    train,
    get_related_document_words,
    get_top_words_per_topic, 
    occurance_counts
)

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(f"--alpha", type=float, default=0.1)
parser.add_argument(f"--beta", type=float, default=0.1)
parser.add_argument(f"--num_topics", type=int, default=10)
parser.add_argument(f"--iterations", type=int,default=100)


args = parser.parse_args()

num_topics = args.num_topics
docs, topic_map, total_words = create_data_set(number_of_topics=num_topics)
str_to_int, int_to_str = create_word_mappings(docs)
co_occurance, occurance, unique = occurance_counts(docs)

n_d_k = get_n_d_k(docs, topic_map, num_topics)
m_k_v = get_m_k_v(docs, topic_map, num_topics, str_to_int)
m_k = m_k_v.sum(axis=1)


topic_map, scores = train(
    args.iterations, total_words, docs, 
    str_to_int, num_topics, topic_map, 
    args.alpha, args.beta, m_k, n_d_k, m_k_v,
    co_occurance, occurance
    
)
related_words = get_related_document_words(docs, topic_map)
common_words_topic = get_top_words_per_topic(related_words)

for k_i in range(args.num_topics):
    print(f"\nThe 20 most common words for topic {k_i} are (score={scores[-1, k_i]}):", flush=True)
    common_words = [word for word,_ in common_words_topic[k_i]]
    print(common_words)

plt.plot(scores.mean(axis=1))
print(scores.mean(axis=1))
plt.savefig(f"mean_umass_alpha={args.alpha}_beta={args.beta}_k={args.num_topics}.png")

