from typing import Tuple, Dict, FrozenSet
from dataclasses import dataclass

Edge = Tuple[str, str]

@dataclass
class RunResult:
    pm: Dict[str, FrozenSet[str]]
    total_score: float