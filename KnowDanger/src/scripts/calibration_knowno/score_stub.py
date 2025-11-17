
"""
Scoring stub for KnowDanger safety. Replace `score_option` with your actual
safety model. The only requirement is that higher scores mean "safer."
"""

from typing import List
import re

def score_option(context: str, option: str) -> float:
    """
    Very naive heuristic to produce a score in [0,1].
    Replace this with your real KnowDanger safety scoring.
    """
    text = f"{context} {option}".lower()

    # Example rule penalties for obviously unsafe patterns (toy only):
    unsafe_keywords = [
        "detonate", "ignite", "explosive", "weapon", "stab",
        "acid", "toxic", "microwave metal", "harm", "injure",
        "disable safety", "override", "bypass", "poison"
    ]
    penalty = 0.0
    for k in unsafe_keywords:
        if k in text:
            penalty += 0.6

    # Example ambiguity penalty
    if any(w in text for w in ["maybe", "not sure", "somehow", "???"]):
        penalty += 0.2

    # Proximity/ethics toy checks
    if "near person" in text or "toward person" in text:
        penalty += 0.2

    # Clamp and invert to make a "safety score"
    score = max(0.0, 1.0 - penalty)
    return min(score, 1.0)


def score_all(context: str, options: List[str]) -> List[float]:
    return [score_option(context, opt) for opt in options]
