
"""
Runtime prediction set + gate for KnowDanger using the calibrated q_hat.
"""

from typing import List, Dict, Any

def prediction_set(scores: List[float], q_hat: float) -> List[int]:
    """
    Given per-option safety scores f(x,y) in [0,1], and calibration q_hat,
    return indices j such that f(x,y_j) >= 1 - q_hat.
    """
    thresh = 1.0 - q_hat
    return [j for j, s in enumerate(scores) if s >= thresh]


def decide(scores: List[float], q_hat: float, hard_safe_mask: List[bool] | None = None) -> Dict[str, Any]:
    """
    Decision wrapper:
      - If |C|==1 and hard safety allows it -> EXECUTE that option.
      - If |C|>1 -> AMBIGUOUS: ask/clarify or rewrite to disambiguate.
      - If |C|==0 -> HALT/ESCALATE.
    Optionally apply a final hard safety mask to veto any option at the last step.
    """
    C = prediction_set(scores, q_hat)
    if hard_safe_mask is not None:
        C = [j for j in C if hard_safe_mask[j]]

    if len(C) == 1:
        return {"policy": "EXECUTE", "choice": C[0], "C": C}
    elif len(C) == 0:
        return {"policy": "HALT", "C": C}
    else:
        return {"policy": "ASK_OR_REWRITE", "C": C}
