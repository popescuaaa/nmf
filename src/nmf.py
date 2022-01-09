from __future__ import division, print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn import decomposition
import json
import os
import numpy as np
import pandas as pd


class NMF:
    """

    Class for NMF model.
    This is a wrapper for sklearn.decomposition.NMF,
    which can perform data preprocessing, data acquisition, and model training.

    """

    def __init__(self,
                 csv_path: str,
                 outdir: str,
                 topics: int = 10,
                 iterations: int = 100,
                 words: int = 10,
                 write_output: bool = True,
                 min_count: int = 10,
                 max_freq: int = 20):

        self.csv_path = csv_path
        self.outdir = outdir
        self.topics = topics
        self.iterations = iterations
        self.words = words
        self.write_output = write_output
        self.min_count = min_count
        self.max_freq = max_freq

        # Set random seed
        np.random.seed(42)

        # build corpus
        self.corpus = self.acquire_data()
        self.corpus = self.corpus.values[:, :3]
        self.corpus = [x[1] for x in self.corpus]
        self.corpus = np.array(self.corpus)

        # build term document matrix
        self.vec = self.get_tfidf(self.topics, self.words)
        self.tdm = self.get_tdm()

        # factor term document matrix
        self.feature_names = self.vec.get_feature_names()
        self.nmf = self.build_nmf()

        # get topics by documents and topics by terms
        self.documents_by_topics = self.get_documents_by_topics()
        self.topics_by_terms = self.nmf.components_

        # get topics by documents and topics by terms
        self.docs_to_topics = self.get_doc_to_topics()
        self.topics_to_words = self.get_topic_to_words()

        # write the results
        if self.write_output:
            self.write_results()

    def write_json(self, filename, obj):
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        with open(os.path.join(self.outdir, filename), 'w') as out:
            json.dump(obj, out)

    def acquire_data(self) -> pd.DataFrame:
        """
            Acquire data from the csv file
        """
        raw_data = pd.read_csv(self.csv_path)
        print(raw_data)
        return raw_data

    def get_tfidf(self, topics, n_words):
        """
            Return a TFIDF for building the input TDM matrix
        """
        return TfidfVectorizer(
            input='content',
            stop_words='english',
            max_df=self.max_freq,
            min_df=self.min_count,
            max_features=topics * n_words * 1000
        )

    def get_tdm(self):
        """
            Return a TDM to factor
        """
        return self.vec.fit_transform(self.corpus)

    def build_nmf(self):
        """
            Build the NMF model
            :return: A sklearn NMF that can be used to factor the TDM
        """
        return decomposition.NMF(n_components=self.topics,
                                 random_state=1,
                                 max_iter=self.iterations)

    def get_documents_by_topics(self):
        np.seterr(divide='ignore', invalid='ignore')
        docs_by_topics = self.nmf.fit_transform(self.tdm)
        normalized = docs_by_topics / np.sum(docs_by_topics, axis=1, keepdims=True)
        return np.nan_to_num(normalized)  # zero out nan's

    def get_doc_to_topics(self):
        """
            Find the distribution of each topic in each document
            In our case, the document is the row from the dataframe
        """

        doc_to_topics = defaultdict(lambda: defaultdict())
        for doc_id, topic_values in enumerate(self.documents_by_topics):
            for topic_id, topic_presence_in_doc in enumerate(topic_values):
                data = self.corpus[doc_id]
                doc_to_topics[data][topic_id] = topic_presence_in_doc
        return doc_to_topics

    def get_topic_to_words(self):
        """
            Find the top words for each topic
        """
        topic_to_words = defaultdict(list)
        for topic_id, topic in enumerate(self.topics_by_terms):
            top_features = topic.argsort()[:-self.words - 1:-1]
            topic_to_words[topic_id] = [self.feature_names[i] for i in top_features]
        return topic_to_words

    def write_results(self):
        self.write_json('doc_to_topics.json', self.docs_to_topics)
        self.write_json('topic_to_words.json', self.topics_to_words)

