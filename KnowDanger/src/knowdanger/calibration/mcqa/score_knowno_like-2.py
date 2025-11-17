
# score_knowno_like.py
# KnowNo-style scorer that uses LLM per-option normalized logits (MCQA).
#
# Expected integration:
#   - Place this file next to your `query_lm.py` (or on PYTHONPATH).
#   - `query_lm.py` must expose at least one of the following callables:
#       * label_logits(context: str, options: List[str]) -> List[float]
#       * mcqa_label_logits(context: str, options: List[str]) -> List[float]
#       * get_label_logits(context: str, options: List[str]) -> List[float]
#       * A class LMScorer with method label_logits(context, options) -> List[float]
#
# Notes:
#   - Calls must be DETERMINISTIC: temperature=0, fixed few-shots/prompts.
#   - We apply softmax to logits to get probabilities in [0,1].
#
from typing import List, Dict, Any, Callable
import importlib
import numpy as np

def softmax(x):
    x = np.array(x, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    z = e.sum()
    return (e / z).tolist() if z > 0 else [1.0/len(x)]*len(x)

# Attempt to discover a usable function/class inside query_lm.py
def _resolve_llm_logits_callable() -> Callable[[str, List[str]], List[float]]:
    mod = importlib.import_module("query_lm")
    # function names we try in order
    fn_names = ["label_logits", "mcqa_label_logits", "get_label_logits"]
    for name in fn_names:
        if hasattr(mod, name):
            fn = getattr(mod, name)
            if callable(fn):
                return fn
    # class LMScorer with method label_logits
    if hasattr(mod, "LMScorer"):
        cls = getattr(mod, "LMScorer")
        inst = cls()
        if hasattr(inst, "label_logits") and callable(inst.label_logits):
            return inst.label_logits
    raise NotImplementedError(
        "query_lm.py must provide one of: label_logits, mcqa_label_logits, get_label_logits, "
        "or LMScorer.label_logits(context, options)."
    )

# Resolve once on import (fail fast if not present)
_LLM_LOGITS_FN = _resolve_llm_logits_callable()

def score_all(context: str, options: List[str], meta: Dict[str, Any] | None = None) -> List[float]:
    """
    Returns per-option probabilities in [0,1], higher = model-preferred (safety replica mode).
    """
    logits = _LLM_LOGITS_FN(context, options)
    return softmax(logits)
