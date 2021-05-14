from typing import List

from pydantic import BaseModel

from .document import Document


class Model(BaseModel):
    documents: List[Document]
    terms: List[str]
