import json
from typing import List, Optional, cast

import numpy as np
from pydantic import ValidationError

from .models import Document, FullDocument, IndexedDocument, Model, Query
from .tokenizer import Tokenizer
from .vectorizer import Vectorizer


class Engine:
    def __init__(self):
        self.documents: List[Document] = []
        self.tokenizer = Tokenizer()
        self.vectorizer = Vectorizer()

    def save(self):
        model = Model(
            documents=self.documents,
            terms=self.vectorizer.terms,
        )
        with open("save/docs_and_terms.json", mode="w+") as file:
            json.dump(model.dict(), file)
        np.save("save/idf.npy", self.vectorizer.idf)
        np.save("save/weights.npy", self.vectorizer.weights)

    def load(self):
        with open("save/docs_and_terms.json") as file:
            data = json.load(file)
            model = Model(**data)
            self.documents = model.documents
            self.tokenizer = Tokenizer()
            self.vectorizer = Vectorizer()
            self.vectorizer.terms = model.terms
        self.vectorizer.idf = np.load("save/idf.npy")
        self.vectorizer.weights = np.load("save/weights.npy")

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
        self.tokenizer = Tokenizer()
        for item in self.documents:
            indexed_documents.append(
                IndexedDocument(
                    **item.dict(),
                    indexes=self.tokenizer.tokenize(f"{item.title} {item.content}"),
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
        query_tokens = self.tokenizer.tokenize(query, remove_stopwords=False)
        results = self.vectorizer.query(list(query_tokens))
        return [
            FullDocument(
                **self.documents[index].dict(),
                index=index,
                relevancy=relevancy,
            )
            for index, relevancy in results[:count]
        ]

    def find_with_feedback(
        self,
        query: str,
        good_feedback: List[int],
        bad_feedback: List[int],
        count: int = 10,
    ) -> List[FullDocument]:
        query_tokens = self.tokenizer.tokenize(query, remove_stopwords=False)
        results = self.vectorizer.query_with_feedback(
            list(query_tokens),
            good_feedback,
            bad_feedback,
        )
        return [
            FullDocument(
                **self.documents[index].dict(),
                index=index,
                relevancy=relevancy,
            )
            for index, relevancy in results[:count]
        ]

    def test_precision(self, filename: str, top: int = 10) -> float:
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
            # print("Result finded")
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
            # print(f"Precision of query {query.id} = {global_precisions[-1]}")
        precision = np.array(global_precisions).mean()
        return cast(float, precision)

    def test_recall(self, filename: str, top: int = 10) -> float:
        queries: List[Query] = []
        with open(filename) as file:
            data = json.load(file)
            for item in data:
                try:
                    query = Query(**item)
                    queries.append(query)
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
        return cast(float, recall)

    def test_f(
        self,
        filename: Optional[str] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        beta: float = 1,
        top: int = 10,
    ) -> float:
        assert filename is not None or (precision is not None and recall is not None)
        p: float = 0
        r: float = 0
        if filename:
            p = self.test_precision(filename, top)
            r = self.test_recall(filename, top)
        else:
            p = cast(float, precision)
            r = cast(float, recall)
        f = (1 + beta * beta) * p * r / (beta * beta * p + r)
        return f

    def test_fallout(self, filename: str, top: int = 10) -> float:
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
        global_fallouts = []
        all_docs_len = len(self.documents)
        for query in queries:
            result = self.find(query.text, top)
            mask = [0 if doc.id in query.docs else 1 for doc in result]
            accu = [mask[0]]
            for i in range(1, len(mask)):
                accu.append(mask[i] + accu[-1])
            local_fallouts: List[float] = []
            docs_len = all_docs_len - len(query.docs)
            for i in range(top):
                # if mask[i]:
                local_fallouts.append(accu[i] / docs_len)
            if local_fallouts:
                fallout = np.array(local_fallouts).mean()
                global_fallouts.append(fallout)
            else:
                global_fallouts.append(0)
        fallout = np.array(global_fallouts).mean()
        return cast(float, fallout)
