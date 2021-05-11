from typing import Optional

from .core.engine import Engine


class Dependencies:
    def __init__(self):
        self.engine: Optional[Engine] = None


dependencies = Dependencies()
