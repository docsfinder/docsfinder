from typing import List, Optional

from pydantic import BaseModel


class Query(BaseModel):
    id: str
    source_id: Optional[str]
    source: Optional[str]
    text: str
    docs: List[str]
