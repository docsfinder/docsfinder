from typing import Optional

from pydantic import BaseModel


class Document(BaseModel):
    id: str
    source_id: Optional[str]
    source: Optional[str]
    title: str
    author: Optional[str]
    content: str
