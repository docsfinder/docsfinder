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

    def query(self, text: List[str], a: float = 0.4) -> List[Tuple[int, float]]:
        q_vect = self.vectorize_query(text, a)
        return self.similarity(q_vect)

    def query_with_feedback(
        self,
        text: List[str],
        good_feedback: List[int],
        bad_feedback: List[int],
        a: float = 0.4,
    ) -> List[Tuple[int, float]]:
        q_vect = self.vectorize_query(text, a)
        fb_vect = self.feedback_rocchio(q_vect, good_feedback, bad_feedback)
        return self.similarity(fb_vect)

    def feedback_rocchio(
        self,
        q_vect: np.ndarray,
        good_feedback: List[int],
        bad_feedback: List[int],
        a: float = 1.0,
        b: float = 0.75,
        y: float = 0.15,
    ) -> np.ndarray:
        good_len = len(good_feedback)
        bad_len = len(bad_feedback)
        sum1 = np.zeros_like(q_vect)
        for index in good_feedback:
            sum1 += self.weights[index]
        sum2 = np.zeros_like(q_vect)
        for index in bad_feedback:
            sum2 += self.weights[index]
        term1 = q_vect * a
        term2 = (b / good_len) * sum1 if good_len else 0
        term3 = (y / bad_len) * sum2 if bad_len else 0
        return term1 + term2 - term3

    def vectorize_query(self, text: List[str], a: float = 0.4) -> np.ndarray:
        freq_vect = np.zeros_like(self.idf)
        for i in range(freq_vect.size):
            freq_vect[i] = text.count(self.terms[i])
        maxn = freq_vect.max()
        q_vect = np.zeros_like(freq_vect)
        for i in range(q_vect.size):
            if freq_vect[i]:
                q_vect[i] = self.idf[i] * (a + (1 - a) * freq_vect[i] / maxn)
            else:
                q_vect[i] = 0
        return q_vect

    def similarity(self, q_vect: np.ndarray) -> List[Tuple[int, float]]:
        ranks = [
            (i, self.cosine_similarity(row, q_vect))
            for i, row in enumerate(self.weights)
        ]
        ranks.sort(key=lambda x: x[1], reverse=True)
        return ranks

    def cosine_similarity(self, doc: np.ndarray, qry: np.ndarray):
        return (
            np.multiply(doc, qry).sum()
            / np.multiply(doc, doc).sum() ** 0.5
            * np.multiply(qry, qry).sum() ** 0.5
        )
