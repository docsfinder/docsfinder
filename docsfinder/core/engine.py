import json
from typing import Dict, List, Tuple

import numpy as np
from pydantic import ValidationError

from .document import Document
from .finder import Finder
from .indexed_document import IndexedDocument
from .indexer import Indexer
from .query import Query
from .vectorizer import Vectorizer


class Engine:
    def __init__(self, filename: str):
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
                    indexes=self.indexer.get_indexes(f"{item.title} {item.content}"),
                ),
            )
        self.vectorizer = Vectorizer(
            [document.indexes for document in self.indexed_documents],
            300,
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

    def test_precision(self, filename: str, top: int = 10):
        queries: List[Query] = []
        with open(filename) as file:
            data = json.load(file)
            for item in data:
                try:
                    query = Query(**item)
                    queries.append(query)
                # except ValidationError as e:
                # print(e)
                except ValidationError:
                    continue
        global_precisions = []
        for query in queries:
            result = self.find(query.text, top)
            mask = [1 if doc.id in query.docs else 0 for (doc, _) in result]
            accu = [mask[0]]
            for i in range(1, len(mask)):
                accu.append(mask[i] + accu[-1])
            local_precisions: List[float] = []
            for i in range(top):
                if mask[i]:
                    local_precisions.append(accu[i] / (i + 1))
            if local_precisions:
                precision = np.array(local_precisions).mean()
                global_precisions.append(precision)
            else:
                global_precisions.append(0)
        precision = np.array(global_precisions).mean()
        return precision

    def test_recall(self, filename: str, top: int = 10):
        queries: List[Query] = []
        with open(filename) as file:
            data = json.load(file)
            for item in data:
                try:
                    query = Query(**item)
                    queries.append(query)
                # except ValidationError as e:
                # print(e)
                except ValidationError:
                    continue
        global_recalls = []
        for query in queries:
            result = self.find(query.text, top)
            mask = [1 if doc.id in query.docs else 0 for (doc, _) in result]
            accu = [mask[0]]
            for i in range(1, len(mask)):
                accu.append(mask[i] + accu[-1])
            local_recalls: List[float] = []
            docs_len = len(query.docs)
            for i in range(top):
                if mask[i]:
                    local_recalls.append(accu[i] / docs_len)
            if local_recalls:
                recall = np.array(local_recalls).mean()
                global_recalls.append(recall)
            else:
                global_recalls.append(0)
        recall = np.array(global_recalls).mean()
        return recall

    def test_f(self, filename: str):
        pass

    def test_fallout(self, filename: str):
        pass
