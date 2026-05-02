from dataclasses import dataclass
from typing import List


@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float]