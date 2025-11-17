
# runtime_gate.py
# Paper-faithful KnowDanger runtime gate using conformal prediction.
#
# What is a "runtime gate"?
# -------------------------
# It's the tiny decision layer between your planner and actuators. Given:
#   - per-option safety scores in [0,1] (higher = safer),
#   - a calibrated artifact with q_hat (so tau := 1 - q_hat),
# it builds a *prediction set* C = { j : score_j >= tau } and applies a
# policy:
#   - |C| == 1 and option passes final hard checks  -> EXECUTE that option
#   - |C| >  1                                      -> ASK_OR_REWRITE
#   - |C| == 0                                      -> HALT (escalate)
#
# This matches the KnowNo decision rule (singleton => act; multi => ask; empty => stop).
#
# API
# ---
# - load_q_hat(artifact_path) -> float
# - prediction_set(scores, q_hat, hard_safe_mask=None) -> List[int]
# - decide(scores, q_hat, ask_threshold=0.0, hard_safe_mask=None,
#          allow_empty_set=True) -> dict
# - decide_with_scorer(context, options, scorer, q_hat, ...) -> dict
# - RuntimeGate class to bundle artifact + settings.
#
# Usage example
# -------------
#   from runtime_gate import RuntimeGate
#   gate = RuntimeGate("/path/calibration_artifact.json", ask_threshold=0.6)
#   scores = scorer.score_all(context, options)          # [0,1] higher = safer
#   result = gate.decide(scores, hard_safe_mask=my_mask) # dict with policy, choice, C, etc.
#
from __future__ import annotations
from typing import List, Optional, Dict, Any, Callable
import json

def load_q_hat(artifact_path: str) -> float:
    with open(artifact_path, "r", encoding="utf-8") as f:
        art = json.load(f)
    if "q_hat" not in art:
        raise KeyError(f"Artifact missing 'q_hat': {artifact_path}")
    return float(art["q_hat"])

def prediction_set(scores: List[float],
                   q_hat: float,
                   hard_safe_mask: Optional[List[bool]] = None) -> List[int]:
    """
    Build the CP prediction set given per-option scores and q_hat.
    Optionally apply a hard safety mask (False => never include that index).
    """
    tau = 1.0 - float(q_hat)
    idxs = [j for j, s in enumerate(scores) if s >= tau]
    if hard_safe_mask is not None:
        idxs = [j for j in idxs if bool(hard_safe_mask[j])]
    return idxs

def decide(scores: List[float],
           q_hat: float,
           ask_threshold: float = 0.0,
           hard_safe_mask: Optional[List[bool]] = None,
           allow_empty_set: bool = True) -> Dict[str, Any]:
    """
    Decide EXECUTE / ASK_OR_REWRITE / HALT using the CP prediction set.
    - ask_threshold: optional extra bar on the *top* score to auto-execute.
    - hard_safe_mask: optional boolean filter (apply final hardware/physics checks).
    - allow_empty_set: if False and C == [], fall back to argmax (legacy behavior).
    Returns a dict with fields: policy, C, choice (if EXECUTE), tau, scores, meta.
    """
    if not scores:
        return {"policy": "HALT", "why": "no-options", "C": [], "scores": []}

    C = prediction_set(scores, q_hat, hard_safe_mask=hard_safe_mask)
    tau = 1.0 - float(q_hat)
    top_idx = max(range(len(scores)), key=lambda j: scores[j])
    top_score = float(scores[top_idx])

    # Empty set policy
    if not C:
        if not allow_empty_set:
            # legacy non-empty behavior: include argmax
            C = [top_idx]
            return {
                "policy": "ASK_OR_REWRITE",
                "why": "empty->argmax (allow_empty_set=False)",
                "C": C, "choice": None, "tau": tau, "scores": scores,
                "meta": {"top_idx": top_idx, "top_score": top_score}
            }
        return {
            "policy": "HALT",
            "why": "empty prediction set",
            "C": [], "choice": None, "tau": tau, "scores": scores,
            "meta": {"top_idx": top_idx, "top_score": top_score}
        }

    # Singleton policy
    if len(C) == 1:
        j = C[0]
        if top_idx == j and top_score >= ask_threshold:
            return {
                "policy": "EXECUTE",
                "why": f"singleton set and top_score >= ask_threshold ({top_score:.3f} >= {ask_threshold:.3f})",
                "C": C, "choice": j, "tau": tau, "scores": scores
            }
        # Singleton but top not confident enough -> ask
        return {
            "policy": "ASK_OR_REWRITE",
            "why": f"singleton but top_score < ask_threshold ({top_score:.3f} < {ask_threshold:.3f})",
            "C": C, "choice": None, "tau": tau, "scores": scores
        }

    # Multi-element set -> ask/clarify or rewrite
    return {
        "policy": "ASK_OR_REWRITE",
        "why": f"set size {len(C)} > 1",
        "C": C, "choice": None, "tau": tau, "scores": scores
    }

def decide_with_scorer(context: str,
                       options: List[str],
                       scorer: Any,
                       q_hat: float,
                       ask_threshold: float = 0.0,
                       hard_safe_mask: Optional[List[bool]] = None,
                       allow_empty_set: bool = True) -> Dict[str, Any]:
    """
    Convenience wrapper when you have a scorer module/class with score_all(context, options).
    """
    if not hasattr(scorer, "score_all"):
        raise AttributeError("scorer must expose score_all(context, options) -> List[float]")
    scores = scorer.score_all(context, options)  # expected in [0,1]
    return decide(scores, q_hat, ask_threshold=ask_threshold,
                  hard_safe_mask=hard_safe_mask, allow_empty_set=allow_empty_set)

class RuntimeGate:
    """
    Small helper that loads q_hat once and applies the CP decision rule repeatedly.
    """
    def __init__(self,
                 calibration_artifact_path: str,
                 ask_threshold: float = 0.0,
                 allow_empty_set: bool = True):
        self.q_hat = load_q_hat(calibration_artifact_path)
        self.ask_threshold = float(ask_threshold)
        self.allow_empty_set = bool(allow_empty_set)

    def decide(self,
               scores: List[float],
               hard_safe_mask: Optional[List[bool]] = None) -> Dict[str, Any]:
        return decide(scores, self.q_hat, ask_threshold=self.ask_threshold,
                      hard_safe_mask=hard_safe_mask, allow_empty_set=self.allow_empty_set)

    def decide_with(self,
                    context: str,
                    options: List[str],
                    scorer: Any,
                    hard_safe_mask: Optional[List[bool]] = None) -> Dict[str, Any]:
        return decide_with_scorer(context, options, scorer, self.q_hat,
                                  ask_threshold=self.ask_threshold,
                                  hard_safe_mask=hard_safe_mask,
                                  allow_empty_set=self.allow_empty_set)
