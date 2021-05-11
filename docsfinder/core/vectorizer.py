from typing import Iterable, List, cast

import numpy as np
from gensim.models import Word2Vec


class Vectorizer:
    def __init__(self, sentences: Iterable[Iterable[str]], size: int = 100):
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=size,
            min_count=2,
            window=5,
            sg=1,
            workers=4,
        )
        self.size = size

    def vectorize(self, doc_tokens: List[str]) -> np.ndarray:
        embeddings = []
        if len(doc_tokens) < 1:
            return np.zeros(self.size)
        else:
            for token in doc_tokens:
                try:
                    embeddings.append(self.model.wv.get_vector(token))
                except KeyError:
                    embeddings.append(np.random.rand(self.size))
            # mean the vectors of individual words to get the vector of the document
            return cast(np.ndarray, np.mean(embeddings, axis=0))
