from collections import Counter
import numpy as np


def init_lda(docs, k_topics, vocab_size):
    '''
    Regarding returns of this function:
    topic_assignments is the z-matrix, i.e.,   topic_assignments[d][j]= z_dj
    document_topics_counts is the n-matrix,    document_topics_counts[d][k] = n_dk
    topic_word_counts is the m-matrix,         topic_word_counts[k][v] = m_kv
    topic_counts is total number of topics assigned to topic k,  topic_counts[k]= sum(m_kv)

    '''
    document_topic_counts = [[0]*k_topics for _ in docs]  # Every document gets assigned k topics (topic zero initially)
    topic_word_counts = [[0]*vocab_size for _ in range(k_topics)] #every topic has a word count of the words in the topic
    topic_counts = [0] * k_topics 
    topic_assignments = [] # will randomly assign a topic k to each word initially, mimics docs

    for d_index, doc in enumerate(docs):
        assignments = []
        for word in doc:
            topic = np.random.randint(k_topics)
            assignments.append(topic)
            document_topic_counts[d_index][topic] += 1
            topic_word_counts[topic][word] += 1
            topic_counts[topic] += 1
        topic_assignments.append(assignments)

    return topic_assignments, document_topic_counts, topic_word_counts, topic_counts


def word_occurence_count(docs, vocab_size):
    document_word_sets = [] 
    for document in docs:
        document_word_sets.append(set(document)) 

    cooccurrence_counts = Counter()
    word_document_counts = Counter() 

    for word_set in document_word_sets:
        for w in word_set:
            word_document_counts[w] += 1
        for w1 in word_set:
            for w2 in word_set:
                if w1 != w2:
                    cooccurrence_counts[(w1, w2)] += 1

    # so cooccurrence holds the counts of which words appear together, {(economy,state):5, (state,economy):5}
    # word_document_counts is the denominator to be used in umass,
    # i.e. how many documents a certain word appears
    return cooccurrence_counts, word_document_counts


def gibbs_sampling(docs, topic_assignments, document_topic_counts,
                                     topic_word_counts, topic_counts, alpha, beta, vocab_size,
                                     n_iterations, int_to_str, cooccurrence_counts,
                                      word_document_counts, top_n=20 ): #just a few inputs (-:

    total_tokens = sum(len(d) for d in docs)
    print(f"The total number of tokens in all documents is equal to {total_tokens}")
    coherence_history = []

    for iteration in range(n_iterations):
        for _ in range(total_tokens):
            sample_one_token(docs, topic_assignments, document_topic_counts,
                             topic_word_counts, topic_counts,alpha, beta, vocab_size)


        phi = compute_phi(topic_word_counts, beta, vocab_size)
        coherence_scores = umass_coherence(phi, int_to_str, cooccurrence_counts,
                                                    word_document_counts, top_n=top_n)

        coherence_history.append((iteration+1, coherence_scores))
        print(f"Iteration {iteration+1}: UMass mean coherence = {np.mean(coherence_scores)}", flush=True)

    return coherence_history

def sample_one_token(docs, topic_assignments, document_topic_counts,
                     topic_word_counts, topic_counts, alpha, beta, vocab_size):
    
    d = np.random.randint(len(docs))
    if len(docs[d]) == 0: #some tiny article from reuters without any tokens...
        #print("Document without tokens")
        return

    i = np.random.randint(len(docs[d]))  #choose on word from the document 
    old_topic = topic_assignments[d][i]  
    w = docs[d][i]

    # theese three lines will rempove the topic, i.e. do the -dj thing, 
    # we will be assigning a new topic to the word (random choice later)
    document_topic_counts[d][old_topic] -= 1 
    topic_word_counts[old_topic][w] -= 1
    topic_counts[old_topic] -= 1

    probs = compute_topic_prob(d, w, document_topic_counts,
                                         topic_word_counts, topic_counts, 
                                        alpha, beta, vocab_size)

    new_topic = np.random.choice(len(probs), p=probs) # 0,..,k_topics and choice of probs. Could be the same as before

    topic_assignments[d][i] = new_topic
    document_topic_counts[d][new_topic] += 1
    topic_word_counts[new_topic][w] += 1
    topic_counts[new_topic] += 1


def compute_topic_prob(document_index, word_id, document_topic_counts,
                               topic_word_counts, topic_counts, alpha, beta, vocab_size):
    k_topics = len(topic_counts)
    probs = np.zeros(k_topics)

    # q_k formula (page 15 in the lecture pdf)
    for k in range(k_topics):
        left = document_topic_counts[document_index][k] + alpha
        right = (topic_word_counts[k][word_id] + beta) 
        denominator = topic_counts[k] + vocab_size*beta
        probs[k] = (left * right) / denominator

    return probs / probs.sum()


def get_top_words_per_topic(phi, int_to_str, top_n=20):
    topic_top_words = []
    for topic_dist in phi:
        top_ids = np.argsort(topic_dist)[::-1][:top_n]
        topic_top_words.append(list(top_ids))
    return topic_top_words


def compute_phi(topic_word_counts, beta, vocab_size):
    phi = []
    for counts in topic_word_counts:
        tot = sum(counts) + vocab_size*beta
        phi.append([(c+beta)/tot for c in counts])

    return phi #returns (k_topics, words) matrix. I.e. probabilities for each word given topic


def umass_coherence(phi, int_to_str, cooccurrence_counts, word_document_counts, top_n=20, epsilon=1):
    topic_indices = get_top_words_per_topic(phi, int_to_str, top_n)
    coherence_scores = []

    for topic_word_ids in topic_indices:
        score = 0
        for j in range(1, len(topic_word_ids)):
            for i in range(j):
                w_j = topic_word_ids[j]
                w_i = topic_word_ids[i]
                numerator = cooccurrence_counts[(w_j, w_i)] + epsilon
                denominator = word_document_counts[w_i]
                score += np.log(numerator / denominator)
        coherence_scores.append(score)

    return coherence_scores