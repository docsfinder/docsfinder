import json
from typing import List

import numpy as np
from pydantic import ValidationError

from .document import Document
from .full_document import FullDocument
from .indexed_document import IndexedDocument
from .indexer import Indexer
from .model import Model
from .query import Query
from .vectorizer import QueryEmptyException, Vectorizer


class Engine:
    def __init__(self):
        self.documents: List[Document] = []
        self.indexer = Indexer()
        self.vectorizer = Vectorizer()

    def save(self):
        model = Model(
            documents=self.documents,
            idf=self.vectorizer.idf,
            n=self.vectorizer.n,
            w=self.vectorizer.w,
            terms=self.vectorizer.terms,
        )
        with open("model.json", mode="w+") as file:
            json.dump(model.dict(), file)

    def load(self):
        with open("model.json") as file:
            data = json.load(file)
            model = Model(**data)
            self.documents = model.documents
            self.indexer = Indexer()
            self.vectorizer = Vectorizer()
            self.vectorizer.n = model.n
            self.vectorizer.idf = model.idf
            self.vectorizer.w = model.w
            self.vectorizer.terms = model.terms

    def train(self, filename: str):
        self.documents: List[Document] = []
        indexed_documents: List[IndexedDocument] = []
        print("Starting engine ...")
        print("Loading data ...")
        with open(filename) as file:
            data = json.load(file)
            for item in data:
                try:
                    document = Document(**item)
                    self.documents.append(document)
                except ValidationError:
                    continue
        print("Data loaded")
        print("Indexing documents ...")
        self.indexer = Indexer()
        for item in self.documents:
            indexed_documents.append(
                IndexedDocument(
                    **item.dict(),
                    indexes=self.indexer.get_indexes(f"{item.title} {item.content}"),
                ),
            )
        print("Documents indexed")
        self.vectorizer = Vectorizer()
        print("Training model ...")
        self.vectorizer.train(
            [document.indexes for document in indexed_documents],
        )
        print("Model trained")

    def find(self, query: str, count: int = 10) -> List[FullDocument]:
        try:
            query_tokens = self.indexer.get_indexes(query, remove_stopwords=False)
            results = self.vectorizer.query(list(query_tokens))
            for index, relevancy in results[:count]:
                yield FullDocument(**self.documents[index].dict(), relevancy=relevancy)
        except QueryEmptyException as e:
            print(query)
            raise e

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
        len_queries = len(queries)
        for index, query in enumerate(queries):
            print(f"Run query {index + 1} of {len_queries}")
            result = self.find(query.text, top)
            mask = [1 if doc.id in query.docs else 0 for doc in result]
            print("Result finded")
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
            mask = [1 if doc.id in query.docs else 0 for doc in result]
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
