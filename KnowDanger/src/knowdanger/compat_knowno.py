
"""
Compatibility shim for various KnowNo (lang-help) layouts.
If your upstream repo exposes different entry points, edit this file and ensure
that `KnownoAdapter` can import these functions by name.

We expect two callables:
    calibrate(alpha: float, score_sets: list[list[float]]) -> float  # returns tau threshold
    predict_set(scores: list[float], tau: float, alpha: float) -> list[int]

Below is an example mapping if the upstream provides a class `Calibrator` and a function `prediction_set`.
"""

# --- EDIT THIS MAPPING TO YOUR UPSTREAM IF NEEDED ---
def calibrate(alpha: float, score_sets: list[list[float]]) -> float:
    # Example: your upstream might have Calibrator.fit(score_sets, alpha) -> tau
    # from lang_help.knowno.calibration import Calibrator
    # return Calibrator(alpha).fit(score_sets)
    # Placeholder fallback:
    import math
    tops = [max(s) if s else 0.0 for s in score_sets]
    q = 1.0 - alpha
    return float(sorted(tops)[max(0, int(math.floor(q * (len(tops)-1))))]) if tops else 0.5

def predict_set(scores: list[float], tau: float, alpha: float) -> list[int]:
    if not scores:
        return []
    idxs = [i for i, v in enumerate(scores) if v >= tau]
    if not idxs:
        idxs = [int(max(range(len(scores)), key=lambda i: scores[i]))]
    return idxs
