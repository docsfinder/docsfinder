from typing import Optional

from pydantic import BaseModel


class Document(BaseModel):
    id: str
    source_id: str
    title: str
    author: Optional[str]
    content: str
