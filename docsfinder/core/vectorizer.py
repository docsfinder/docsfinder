from math import log10
from typing import List, Tuple

import numpy as np


class Vectorizer:
    def __init__(self):
        self.idf: np.ndarray
        self.terms: List[str]
        self.weights: np.ndarray

    def train(self, docs: List[List[str]]):
        terms = [item for item in {term for doc in docs for term in doc}]
        len_docs = len(docs)
        len_terms = len(terms)
        freq_vect = np.zeros((len_docs, len_terms))
        for i in range(freq_vect.shape[0]):
            for j in range(freq_vect.shape[1]):
                freq_vect[i, j] = docs[i].count(terms[j])
        max_vect = np.amax(freq_vect, 1)
        f_vect = (freq_vect.T / max_vect).T
        idf = np.vectorize(lambda x: log10(len_docs / x))(
            np.count_nonzero(freq_vect, 0)
        )
        for i in range(f_vect.shape[0]):
            f_vect[i, :] *= idf
        self.idf = idf
        self.terms = terms
        self.weights = f_vect

    def query(self, document: List[str], a: float = 0.4) -> List[Tuple[int, float]]:
        freq_vect = np.zeros_like(self.idf)
        for i in range(freq_vect.size):
            freq_vect[i] = document.count(self.terms[i])
        maxn = freq_vect.max()
        q_vect = np.zeros_like(freq_vect)
        for i in range(q_vect.size):
            if freq_vect[i]:
                q_vect[i] = self.idf[i] * (a + (1 - a) * freq_vect[i] / maxn)
            else:
                q_vect[i] = 0
        ranks = [(i, self.rankf(row, q_vect)) for i, row in enumerate(self.weights)]
        ranks.sort(key=lambda x: x[1], reverse=True)
        return ranks

    def rankf(self, doc: np.ndarray, qry: np.ndarray):
        return (
            np.multiply(doc, qry).sum()
            / np.multiply(doc, doc).sum() ** 0.5
            * np.multiply(qry, qry).sum() ** 0.5
        )
