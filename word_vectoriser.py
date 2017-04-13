from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
from random import sample

class Word_Vectoriser(object):

    def __init__(self, w2v):
        self.w2v = w2v
        self.dim = len(w2v.word_vec('case'))
        self.word2weight = None

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def get_params(self, *args, **kwargs):
        return self

    def transform(self, X):
        try:
            return np.array(sample([np.mean([self.w2v.word_vec(w) * self.word2weight[w]
                for w in words if w in self.w2v.index2word]
                     or [np.zeros(self.dim)], axis=0)
                         for words in X], 4)).ravel()
        except Exception as e:
            print(e)
            return np.array([])


    """
    def transform(self, X):
        try:
            return np.array(sample([np.mean([self.w2v.word_vec(w) * self.word2weight[w]
                for w in words if w in self.w2v.index2word]
                     or [np.zeros(self.dim)], axis=0)
                         for words in X], 10)).ravel()
        except Exception as e:
            print(e)
    """
    
