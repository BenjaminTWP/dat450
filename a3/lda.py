import numpy as np

def lda(n_docs, vocab_size, alpha=0.01, beta=0.01,  k_topics=20):

    theta_docs = np.random.dirichlet([alpha]*k_topics, size=n_docs)

    phi_topics = np.random.dirichlet([beta]*vocab_size, size=k_topics)

    return theta_docs, phi_topics

