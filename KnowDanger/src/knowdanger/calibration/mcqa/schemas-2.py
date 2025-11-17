
# schemas.py
# Minimal data structures & helpers shared by tools.

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class MCQAExample:
    """
    A single MCQA-style example for KnowDanger safety calibration.
    - id: unique string id (e.g., "break_0001")
    - context: short scene/instruction description shown to the model/labeler
    - options: list of candidate actions (strings)
    - scores: optional per-option safety/confidence scores in [0,1] (higher = safer)
    - meta: optional extra info (scene family, seed, etc.); should be JSON-serializable
    """
    id: str
    context: str
    options: List[str]
    scores: Optional[List[float]] = None
    meta: Optional[Dict[str, Any]] = None
