from typing import List, Optional

from pydantic import BaseModel


class Document(BaseModel):
    id: str
    source_id: Optional[str]
    source: Optional[str]
    title: str
    author: Optional[str]
    content: str


class FullDocument(Document):
    index: int
    relevancy: float


class IndexedDocument(Document):
    indexes: List[str]


class Model(BaseModel):
    documents: List[Document]
    terms: List[str]


class Query(BaseModel):
    id: str
    source_id: Optional[str]
    source: Optional[str]
    text: str
    docs: List[str]
