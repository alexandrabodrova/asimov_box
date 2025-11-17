
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Example:
    """
    One MCQA-style safety example for calibration/evaluation.
    - id: unique identifier
    - context: human-readable description of the scene/instruction
    - options: candidate actions (A, B, C, ... are implicit by index)
    - safe_set: indices of safe options (can be empty at calibration time;
                you'll fill these with the labeling CLI)
    - scores: optional precomputed safety scores for each option in [0,1].
              If missing, the label CLI can compute scores via a scoring function.
    - meta: arbitrary metadata (scene type, seed, etc.).
    """
    id: str
    context: str
    options: List[str]
    safe_set: List[int]
    scores: Optional[List[float]] = None
    meta: Optional[Dict[str, Any]] = None
