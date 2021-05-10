import json
from typing import Dict, List, Tuple

import numpy as np
from pydantic import ValidationError

from .document import Document
from .finder import Finder
from .indexed_document import IndexedDocument
from .indexer import Indexer
from .vectorizer import Vectorizer


class Engine:
    def __init__(self, filename: str = "data.json"):
        self.documents: List[Document] = []
        self.indexed_documents: List[IndexedDocument] = []
        self.doc_vectors: Dict[str, np.ndarray] = {}
        with open(filename) as file:
            data = json.load(file)
            for item in data:
                try:
                    document = Document(**item)
                    self.documents.append(document)
                except ValidationError:
                    continue
        self.indexer = Indexer()
        for item in self.documents:
            self.indexed_documents.append(
                IndexedDocument(
                    **item.dict(),
                    indexes=self.indexer.get_indexes(item.content),
                ),
            )
        self.vectorizer = Vectorizer(
            [document.indexes for document in self.indexed_documents],
        )
        for document in self.indexed_documents:
            self.doc_vectors[document.id] = self.vectorizer.vectorize(document.indexes)
        self.finder = Finder(self.doc_vectors)

    def find(self, query: str, count: int = 10) -> List[Tuple[Document, float]]:
        query_tokens = self.indexer.get_indexes(query, remove_stopwords=False)
        query_vector = self.vectorizer.vectorize(list(query_tokens))
        results = self.finder.find(query_vector)
        dict_documents = {document.id: document for document in self.documents}
        for result in results[:count]:
            yield (dict_documents[result[0]], result[1])
