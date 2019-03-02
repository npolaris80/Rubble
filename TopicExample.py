import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

def display_topics(model, feature_names, no_top_words, display_output=True):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        if display_output:
            print("Topic %d:" % (topic_idx))
            print(" ".join(topic_words))
        topic_dict[topic_idx] = topic_words

    return topic_dict

def create_corpus(dir):
    documents = []
    for entry in os.scandir(dir):
        if not entry.name.startswith('.') and entry.is_file():
            # print(entry.name)
            with open(dir+'/'+entry.name,'r') as f:
                fread = f.read()
            documents.append(fread)
    return documents

def run_nmf(documents, num_features, num_topics):
    # NMF is able to use tf-idf
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    # Run NMF/ get topics
    nmf_decomp = NMF(n_components=num_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    return nmf_decomp, tfidf_feature_names

def run_lda(documents, num_features, num_topics):
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    # Run LDA
    lda_decomp = LatentDirichletAllocation(n_topics=num_topics, max_iter=5, learning_method='online',
                                           learning_offset=50., random_state=0).fit(tf)
    return lda_decomp, tfidf_feature_names

#parameters
dir = './data'
documents = create_corpus(dir)

num_features = 100
num_topics = 3
num_top_words = 10


nmf_decomp, tfidf_feature_names = run_nmf(documents, num_features, num_topics)
nmf_topics = display_topics(nmf_decomp, tfidf_feature_names, num_top_words)

lda_decomp, tf_feature_names =run_lda(documents, num_features, num_topics)
lda_topics = display_topics(lda_decomp, tf_feature_names, num_top_words)


#TODO: Use PMI code (Gensim?) to select most coherent topics?