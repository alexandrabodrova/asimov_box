
"""
Paper-faithful KnowNo adapters (strict)
=======================================
Two adapters:
  1) ChoiceBaseline: reproduces the original KnowNo semantics (action-choice uncertainty only).
     SAFE iff |prediction_set(scores)| == 1; otherwise UNCERTAIN. No thresholds, no top-1 fallback.
  2) SafetyAdapter: uses the *same* upstream predictor but applies it to a binary SAFE/UNSAFE score vector
     per step to assess safety-of-action in KnowDanger fusion:
        - Provide either Step.meta["safety_scores"] = [p_safe, p_unsafe] (probs or logits)
          or Step.meta["safety_candidates"] = [("safe", score), ("unsafe", score)].
        - Prediction set S over {safe, unsafe}:
              S = {"safe"}     -> SAFE
              S = {"unsafe"}   -> UNSAFE
              S = {"safe","unsafe"} or empty -> UNCERTAIN
Both adapters REQUIRE upstream KnowNo to be importable (agent.predict.*). No shims here.
"""
from __future__ import annotations
import importlib, os, sys, inspect, types
from typing import List, Tuple, Optional, Any

class _UpstreamCP:
    def __init__(self, alpha: float = 0.1, cfg_json: Optional[str] = None):
        # Make upstream importable if KNOWNO_ROOT provided
        kn_root = os.environ.get("KNOWNO_ROOT", "")
        if kn_root and os.path.isdir(kn_root) and kn_root not in sys.path:
            sys.path.append(kn_root)
        tried = [
            "agent.predict.conformal_predictor",
            "agent.predict.multi_step_conformal_predictor"
        ]
        last_err = None
        self.cp_mod = None
        self.cp_cls = None
        for modname in tried:
            try:
                mod = importlib.import_module(modname)
                cls = getattr(mod, "ConformalPredictor", None) or getattr(mod, "MultiStepConformalPredictor", None)
                if cls is not None:
                    self.cp_mod = mod; self.cp_cls = cls; break
            except Exception as e:
                last_err = e
        if self.cp_cls is None:
            raise RuntimeError(f"Cannot import upstream KnowNo predictor from {tried}: {last_err}")

        self.cp = self._instantiate(self.cp_cls, alpha, cfg_json)
        self.tau = None

    @staticmethod
    def _instantiate(cp_cls, alpha: float, cfg_json: Optional[str]):
        cfg_obj = None; cfg_dict = None
        if cfg_json:
            import json
            with open(cfg_json, "r") as f:
                cfg_dict = json.load(f)
            # try util config types
            util = None
            for cand in ("agent.predict.util", "agent.predict.config"):
                try:
                    import importlib as _il
                    util = _il.import_module(cand); break
                except Exception: pass
            if util:
                for name in ("Config","PredictorConfig","CPConfig","ConformalConfig"):
                    C = getattr(util, name, None)
                    if inspect.isclass(C):
                        try:
                            cfg_obj = C(**cfg_dict); break
                        except Exception:
                            try:
                                cfg_obj = C()
                                for k,v in cfg_dict.items():
                                    try: setattr(cfg_obj, k, v)
                                    except Exception: pass
                                break
                            except Exception:
                                pass
            if cfg_obj is None:
                cfg_obj = types.SimpleNamespace(**cfg_dict)

        trials = [
            ("alpha_kw", lambda: cp_cls(alpha=alpha)),
            ("empty",    lambda: cp_cls()),
        ]
        if cfg_obj is not None:
            trials += [("cfg_pos", lambda: cp_cls(cfg_obj)), ("cfg_kw", lambda: cp_cls(cfg=cfg_obj))]
        if cfg_dict is not None:
            trials += [("cfg_dict_kwargs", lambda: cp_cls(**cfg_dict))]

        last_err = None
        for _, ctor in trials:
            try:
                return ctor()
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Could not instantiate upstream KnowNo predictor; last error: {last_err}")

    def calibrate(self, score_sets: List[List[float]], alpha: float):
        tau = None
        for m in ("calibrate","fit","train"):
            if hasattr(self.cp, m):
                fn = getattr(self.cp, m)
                try:
                    out = fn(score_sets, alpha)
                except TypeError:
                    try: out = fn(score_sets)
                    except TypeError: out = fn()
                if isinstance(out, (int, float)):
                    tau = float(out)
                elif hasattr(self.cp, "tau") and isinstance(self.cp.tau, (int, float)):
                    tau = float(self.cp.tau)
                break
        self.tau = tau
        return tau

    def predict_set(self, scores: List[float]):
        for m in ("predict_set","predict"):
            if hasattr(self.cp, m):
                fn = getattr(self.cp, m)
                try:
                    out = fn(scores)
                except TypeError:
                    out = fn(scores, self.tau)
                return out["indices"] if isinstance(out, dict) and "indices" in out else out
        raise RuntimeError("Upstream KnowNo predictor lacks predict/predict_set.")

class ChoiceBaseline:
    """Paper-faithful KnowNo baseline: action-choice uncertainty only."""
    def __init__(self, alpha: float = 0.1, cfg_json: Optional[str] = None):
        self.up = _UpstreamCP(alpha=alpha, cfg_json=cfg_json)
        self.alpha = alpha

    def calibrate(self, score_sets: List[List[float]]):
        return self.up.calibrate(score_sets, self.alpha)

    def step_label(self, scores: List[float]) -> Tuple[str, dict]:
        idxs = self.up.predict_set(scores)
        if len(idxs) == 1:
            return "SAFE", {"prediction_set": list(map(int, idxs)), "semantics": "choice"}
        else:
            return "UNCERTAIN", {"prediction_set": list(map(int, idxs)), "semantics": "choice"}

class SafetyAdapter:
    """
    Safety-of-action via CP on binary SAFE/UNSAFE.
    Expects per step one of:
      - safety_scores: [p_safe, p_unsafe] (probs or logits)
      - safety_candidates: [("safe", s1), ("unsafe", s2)] (order not required)
    """
    def __init__(self, alpha: float = 0.1, cfg_json: Optional[str] = None):
        self.up = _UpstreamCP(alpha=alpha, cfg_json=cfg_json)
        self.alpha = alpha

    def calibrate(self, score_sets: List[List[float]]):
        # Each set must be length 2 for binary safety (safe/unsafe)
        return self.up.calibrate(score_sets, self.alpha)

    @staticmethod
    def _extract_binary(meta: dict) -> List[float]:
        if not meta: return None
        if "safety_scores" in meta and isinstance(meta["safety_scores"], (list, tuple)) and len(meta["safety_scores"]) == 2:
            return [float(meta["safety_scores"][0]), float(meta["safety_scores"][1])]
        if "safety_candidates" in meta:
            c = meta["safety_candidates"]
            d = {str(k).lower(): float(v) for k, v in c}
            return [d.get("safe", None), d.get("unsafe", None)]
        return None

    def step_label(self, meta: dict) -> Tuple[str, dict]:
        scores = self._extract_binary(meta)
        if scores is None or any(v is None for v in scores):
            return "UNCERTAIN", {"why": "no safety scores"}
        idxs = self.up.predict_set(scores)  # indices over [safe, unsafe]
        s = set(int(i) for i in idxs)
        if s == {0}:
            return "SAFE", {"prediction_set": [0], "classes": ["safe","unsafe"], "semantics": "safety"}
        elif s == {1}:
            return "UNSAFE", {"prediction_set": [1], "classes": ["safe","unsafe"], "semantics": "safety"}
        else:
            return "UNCERTAIN", {"prediction_set": sorted(list(s)), "classes": ["safe","unsafe"], "semantics": "safety"}
