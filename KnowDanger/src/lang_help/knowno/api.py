# src/lang_help/knowno/api.py
import math, os, sys
from typing import List, Tuple

# Attempt to locate the upstream repo if KNOWNO_ROOT is set (or adjust paths as needed)
ROOTS = [os.environ.get("KNOWNO_ROOT", ""), os.getcwd()]
for r in ROOTS:
    if r and os.path.isdir(r) and r not in sys.path:
        sys.path.append(r)

# Try the upstream predictor(s)
_CP_CLASS = None
for cand in (
    "agent.predict.conformal_predictor",            # your repo layout
    "agent.predict.multi_step_conformal_predictor", # variant
):
    try:
        mod = __import__(cand, fromlist=["*"])
        _CP_CLASS = getattr(mod, "ConformalPredictor", None) or getattr(mod, "MultiStepConformalPredictor", None)
        if _CP_CLASS:
            break
    except Exception:
        pass

def _softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    tot = sum(exps)
    return [e / tot for e in exps] if tot else [1.0/len(scores)]*len(scores)

def _fallback_tau(cal_sets: List[List[float]], alpha: float) -> float:
    tops = []
    for s in cal_sets:
        s = list(s)
        if any(v < 0 or v > 1 for v in s):
            s = _softmax(s)
        tops.append(max(s) if s else 0.0)
    if not tops:
        return 0.5
    q = 1.0 - alpha
    idx = max(0, int(math.floor(q * (len(tops) - 1))))
    return float(sorted(tops)[idx])

# --- Public API expected by KnowDanger adapter ---

def calibrate(alpha: float, score_sets: List[List[float]]) -> float:
    """
    Return a conformal threshold tau. Prefer upstream class if available; else quantile fallback.
    """
    if _CP_CLASS is not None:
        try:
            try:
                cp = _CP_CLASS(alpha=alpha)
            except TypeError:
                cp = _CP_CLASS()
            for m in ("calibrate", "fit", "train"):
                if hasattr(cp, m):
                    out = getattr(cp, m)(score_sets) if m != "calibrate" else getattr(cp, m)(score_sets)
                    if isinstance(out, (int, float)):  # some implementations return tau
                        return float(out)
                    if hasattr(cp, "tau") and isinstance(cp.tau, (int, float)):
                        return float(cp.tau)
                    break
        except Exception:
            pass
    # Fallback: (1 - alpha) quantile of top-scores
    return _fallback_tau(score_sets, alpha)

def predict_set(scores: List[float], tau: float = None, alpha: float = 0.1) -> List[int]:
    """
    Return indices of the prediction set for this score vector.
    """
    s = list(scores)
    if any(v < 0 or v > 1 for v in s):
        s = _softmax(s)
    if _CP_CLASS is not None:
        try:
            try:
                cp = _CP_CLASS(alpha=alpha)
            except TypeError:
                cp = _CP_CLASS()
            # Prefer predict_set; else predict(scores, tau)
            if hasattr(cp, "predict_set"):
                out = cp.predict_set(s)
            else:
                out = cp.predict(s, tau)
            if isinstance(out, dict) and "indices" in out:
                return list(map(int, out["indices"]))
            return list(map(int, out))
        except Exception:
            pass
    # Fallback: threshold on scores; ensure non-empty by taking top-1
    thr = tau if tau is not None else (1.0 - alpha)
    idxs = [i for i, v in enumerate(s) if v >= thr]
    if not idxs:
        idxs = [int(max(range(len(s)), key=lambda i: s[i]))]
    return idxs
