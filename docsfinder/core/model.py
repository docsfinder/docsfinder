from typing import Dict, List

from pydantic import BaseModel

from .document import Document
from .indexed_document import IndexedDocument


class Model(BaseModel):
    documents: List[Document]
    indexed_documents: List[IndexedDocument]
    doc_vectors: Dict[str, List[float]]
