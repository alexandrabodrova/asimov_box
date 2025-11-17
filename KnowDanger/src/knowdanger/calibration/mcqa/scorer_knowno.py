
# score_knowno_like.py
# Replica-mode scorer: LLM-like MCQA probabilities (normalized logits).
# This file provides 2 functions:
#   - softmax(logits) -> probabilities
#   - score_all(context, options, meta) -> List[float]
#
# IMPORTANT:
#  - Replace `llm_label_logits` with a real call into your LLM pipeline that
#    returns per-label logits for options A/B/C/... (deterministic! temperature=0).
#  - If you don't have LLM logits handy, you can temporarily use
#    dummy_label_logits() which makes a deterministic score from text features.

from typing import List, Dict
import math
import numpy as np

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def softmax(x):
    x = np.array(x, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    z = e.sum()
    return (e / z).tolist() if z > 0 else [1.0/len(x)]*len(x)

def llm_label_logits(context: str, options: List[str]) -> List[float]:
    raise NotImplementedError("Wire your LLM here (normalized logprobs over options).")

def dummy_label_logits(context: str, options: List[str]) -> List[float]:
    unsafe = [
        "microwave metal", "ignite", "detonate", "bypass safety", "harm",
        "stab", "acid", "toxic", "toward person", "disable interlock",
        "spill solvent", "overheat", "point laser at eye"
    ]
    safeish = ["place on table", "put down", "move away", "turn off", "ask human"]
    logits = []
    for opt in options:
        t = (context + " " + opt).lower()
        logit = 0.0
        if any(k in t for k in unsafe):
            logit -= 6.0
        if any(k in t for k in ["maybe", "somehow", "?", "not sure"]):
            logit -= 1.0
        if any(k in t for k in safeish):
            logit += 2.0
        logit += max(-1.5, 1.5 - 0.05*len(opt))
        logits.append(logit)
    return logits

def score_all(context: str, options: List[str], meta: Dict | None = None) -> List[float]:
    try:
        logits = llm_label_logits(context, options)
    except NotImplementedError:
        logits = dummy_label_logits(context, options)
    return softmax(logits)
