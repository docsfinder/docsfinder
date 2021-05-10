from typing import Iterable, List, cast

import numpy as np
from gensim.models import Word2Vec


class Vectorizer:
    def __init__(self, sentences: Iterable[Iterable[str]]):
        self.model = Word2Vec(sentences=sentences)

    def vectorize(self, doc_tokens: List[str], size: int = 100) -> np.ndarray:
        embeddings = []
        if len(doc_tokens) < 1:
            return np.zeros(size)
        else:
            for token in doc_tokens:
                try:
                    embeddings.append(self.model.wv.get_vector(token))
                except KeyError:
                    embeddings.append(np.random.rand(size))
            # mean the vectors of individual words to get the vector of the document
            return cast(np.ndarray, np.mean(embeddings, axis=0))
