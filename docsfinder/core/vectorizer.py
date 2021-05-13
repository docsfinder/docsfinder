from math import log10
from typing import List, Tuple

# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity


class QueryEmptyException(Exception):
    pass


class Vectorizer:
    def __init__(self):
        self.w: List[List[float]] = []
        self.n: int = 0
        self.idf: List[float] = []
        self.terms: List[str] = []

    def train(self, sentences: List[List[str]]) -> List[List[float]]:
        self.terms = [
            item for item in {term for sentence in sentences for term in sentence}
        ]
        freq_vect = [
            [sentence.count(term) for term in self.terms] for sentence in sentences
        ]
        max_vect = [max(row) for row in freq_vect]
        f_vect = [[cel / maxn for cel in row] for maxn, row in zip(max_vect, freq_vect)]
        N = len(sentences)
        self.n = len(self.terms)
        self.idf = [
            log10(N / len([1 for row in freq_vect if row[i] > 0]))
            for i in range(self.n)
        ]
        self.w = [[cell * idf for cell, idf in zip(row, self.idf)] for row in f_vect]
        return self.w

    def query(self, sentence: List[str], a: float = 0.4) -> List[Tuple[int, float]]:
        freq_vect = [sentence.count(term) for term in self.terms]
        # print("Query freq_vect ready")
        maxn = max(freq_vect)
        # print("Query maxn ready")
        q_vect = [
            self.idf[i] * (a + (1 - a) * freq_vect[i] / maxn) for i in range(self.n)
        ]
        # print("Query q_vect ready")
        if not sum(q_vect):
            raise QueryEmptyException()
        ranks = [
            # (
            #     i,
            #     cosine_similarity(
            #         np.array(row).reshape(1, -1),
            #         np.array(q_vect).reshape(1, -1),
            #     ).item(),
            # )
            (i, self.rankf(row, q_vect))
            for i, row in enumerate(self.w)
        ]
        # print("Query ranks ready")
        ranks.sort(key=lambda x: x[1], reverse=True)
        return ranks

    def rankf(self, doc: List[float], qry: List[float]):
        assert len(doc) == len(qry)
        return (
            sum(wj * wq for wj, wq in zip(doc, qry))
            / sum(wj * wj for wj in doc) ** 0.5
            * sum(wq * wq for wq in qry) ** 0.5
        )
