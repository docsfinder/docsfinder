from typing import List

from pydantic import BaseModel

from .document import Document


class Model(BaseModel):
    documents: List[Document]
    w: List[List[float]]
    n: int
    idf: List[float]
    terms: List[str]
