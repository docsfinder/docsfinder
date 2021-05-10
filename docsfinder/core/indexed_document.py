from typing import List

from .document import Document


class IndexedDocument(Document):
    indexes: List[str]
