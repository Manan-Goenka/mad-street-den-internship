import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
import numpy as np
import string

wnl = WordNetLemmatizer()


def tokenize_sentence(input_str):
    input_str = input_str.translate(None, string.punctuation)
    input_str = input_str.decode('utf-8')
    tokenized_str = [wnl.lemmatize(unicode(i)).lower() for i in input_str.split(' ')]
    return tokenized_str


def train_model_word2vec(input_file, model_file):
    df = pd.read_csv(input_file, header=None, names=['title'])
    tokenized_corpus = [tokenize_sentence(document) for document in df.title.tolist()]

    # Set values for various parameters
    feature_size = 300  # Word vector dimensionality
    window_context = 10  # Context window size
    min_word_count = 1  # Minimum word count
    sample = 1e-3  # Downsample setting for frequent words

    w2v_model = Word2Vec(tokenized_corpus, size=feature_size,
                         window=window_context, min_count=min_word_count,
                         sample=sample, iter=100)
    w2v_model.save(open(model_file, 'w'))
    del df
    return



def load_model_and_corpus(input_file, model_file):
    #df = pd.read_csv(input_file, header=None, names=['title', 'description', 'gender'])
    df = pd.read_csv(input_file, header=None, names=['title'])
    tokenized_corpus = [tokenize_sentence(document) for document in df.title.tolist()]
    w2v_model = Word2Vec.load(model_file)
    return tokenized_corpus, w2v_model


# view similar words based on gensim's model
# similar_words = {search_term: [item[0] for item in w2v_model.wv.most_similar([search_term], topn=5)]
#                          for search_term in ['bra']}
# print similar_words

def cos_sim(a, b):
	"""Takes 2 vectors a, b and returns the cosine similarity percentage according
	to the definition of the dot product
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)


def visualize(model_file, words):
    w2v_model = Word2Vec.load(model_file)
    similar_words = {search_term: [item[0] for item in w2v_model.wv.most_similar([search_term], topn=20)]
                     for search_term in words}
    similar_words

    words = sum([[k] + v for k, v in similar_words.items()], [])
    wvs = w2v_model.wv[words]

    print wvs
    tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=2)
    np.set_printoptions(suppress=True)
    T = tsne.fit_transform(wvs)
    labels = words

    plt.figure(figsize=(14, 8))
    plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
    for label, x, y in zip(labels, T[:, 0], T[:, 1]):
        plt.annotate(label, xy=(x + 1, y + 1), xytext=(0, 0), textcoords='offset points')
    plt.show()


def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.
    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector


def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in corpus]
    return np.array(features)


def cluster_words(tokenized_corpus, w2v_model, feature_size):
    # get document level embeddings
    w2v_feature_array = averaged_word_vectorizer(corpus=tokenized_corpus, model=w2v_model,
                                                 num_features=feature_size)
    print "Finished basic compute"
    # print(w2v_feature_array)
    corpus_df = pd.DataFrame(w2v_feature_array)
    # print(corpus_df)
    ap = AffinityPropagation()
    ap.fit(w2v_feature_array)
    cluster_labels = ap.labels_
    # print(cluster_labels)
    cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
    # print(cluster_labels)
    final_df = pd.concat([corpus_df, cluster_labels], axis=1)
    final_df.to_csv('check_clustering.csv', header=True, index=False)

    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2, random_state=0)
    # pcs = pca.fit_transform(w2v_feature_array)
    # labels = ap.labels_
    # categories = list(final_df['ClusterLabel'])
    # print(categories)
    # plt.figure(figsize=(8, 6))
    #
    # for i in range(len(labels)):
    #     label = labels[i]
    #     color = 'orange' if label == 0 else 'blue' if label == 1 else 'green'
    #     annotation_label = categories[i]
    #     x, y = pcs[i]
    #     plt.scatter(x, y, c=color, edgecolors='k')
    #     plt.annotate(annotation_label, xy=(x+1e-4, y+1e-3), xytext=(0, 0), textcoords='offset points')
    print (final_df)


if __name__ == "__main__":
    tokenized_corpus, w2v_model = load_model_and_corpus('000', 'Word2Vecmodel3.dat')
    cluster_words(tokenized_corpus, w2v_model, 300)
