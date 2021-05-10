from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Finder:
    def __init__(self, doc_vectors: Dict[str, np.ndarray]):
        self.doc_vectors = doc_vectors

    def find(self, query_vector: np.ndarray) -> List[Tuple[str, float]]:
        similarities = [
            (
                id,
                cosine_similarity(
                    np.array(query_vector).reshape(1, -1),
                    np.array(vector).reshape(1, -1),
                ).item(),
            )
            for (id, vector) in self.doc_vectors.items()
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
