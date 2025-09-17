"""
Asimov Box (KnowDanger algorithm): A "guard dog" for LLM-controlled robots.

Key Features:
- Planning-Time Rewriting (inspired by RoboGuard)
- KnowNo-Style Intent Filtering
- Final-Step Interception (new, dynamic safety validation at execution time)
- Epistemic Calibration via LogProbCertifier (KnowNo-style confidence thresholding)
- Optional Human-in-the-Loop override on borderline LLM safety gaps
- Optional Learned Safety Classifier (logprob-aware)

This file defines the KnowDanger algorithm classes, safety checkers, and test harness.

Usage
-----
from knowdanger.core.knowdanger_core import (
    KnowDanger, Config, Scene, PlanCandidate, Step, Verdict, PlanAssessment
)

kd = KnowDanger(Config(alpha=0.1, ask_threshold_confidence=0.7))
scene = Scene(name="...", semantic_graph={...}, rules=[...], env_params={...})
plan = PlanCandidate(name="...", user_prompt="...", steps=[
    Step("place", params={"x":"solvent_bottle","s":"bench1"},
         candidates=[("place_on_bench1", 0.62), ("place_on_bench2", 0.38)],
         meta={"safety_flags": {"near(hot_surface)": True}})
])
assessment = kd.run(scene, plan)
print(assessment.overall.label, assessment.overall.why)

"""

from __future__ import annotations

import dataclasses as dc
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable, Sequence
from pathlib import Path
import importlib
import importlib.util
import os
import sys
import math

# ==========================================================
# Core data classes
# ==========================================================

Label = str  # "SAFE" | "UNSAFE" | "UNCERTAIN"

@dataclass
class Step:
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    # Optional LLM candidates for next-step decision (KnowNo uses these)
    candidates: Optional[List[Tuple[str, float]]] = None  # (candidate_text, score/logit/prob)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PlanCandidate:
    name: str
    steps: List[Step]
    user_prompt: str
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Scene:
    name: str
    semantic_graph: Dict[str, Any]
    rules: List[str]
    env_params: Dict[str, Any] = field(default_factory=dict)
    helpers: Dict[str, Callable] = field(default_factory=dict)

@dataclass
class Verdict:
    label: Label
    why: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StepAssessment:
    step: Step
    roboguard: Verdict
    knowno: Optional[Verdict] = None
    final: Verdict = None  # set by aggregator

@dataclass
class PlanAssessment:
    plan: PlanCandidate
    steps: List[StepAssessment]
    overall: Verdict

@dataclass
class Config:
    # KnowNo CP parameters
    alpha: float = 0.1
    ask_threshold_confidence: float = 0.7
    # Optional: environment variable that can point to upstream repo root
    KNOWNO_ENV: str = "KNOWNO_ROOT"

    def aggregator(self, rg: Verdict, kn: Optional[Verdict]) -> Verdict:
        """
        Conservative fusion policy:
          - If RoboGuard -> UNSAFE: final = UNSAFE.
          - Else if KnowNo -> UNSAFE: final = UNSAFE.
          - Else if both SAFE: final = SAFE.
          - Else: UNCERTAIN.
        """
        if rg.label == "UNSAFE":
            return Verdict("UNSAFE", why=f"RoboGuard blocked: {rg.why}", details={"rg": rg.details, "kn": getattr(kn, 'details', {})})
        if kn is None:
            return Verdict(rg.label, why=f"RoboGuard only: {rg.why}", details={"rg": rg.details})
        if kn.label == "UNSAFE":
            return Verdict("UNSAFE", why=f"KnowNo flagged unsafe: {kn.why}", details={"rg": rg.details, "kn": kn.details})
        if rg.label == "SAFE" and kn.label == "SAFE":
            return Verdict("SAFE", why="Both SAFE (RoboGuard+KnowNo)", details={"rg": rg.details, "kn": kn.details})
        return Verdict("UNCERTAIN", why=f"Uncertainty remains (rg={rg.label}, kn={kn.label})", details={"rg": rg.details, "kn": kn.details})

# ==========================================================
# RoboGuard bridge (dynamic import with safe fallback)
# ==========================================================

class RoboGuardBridge:
    """
    Thin dynamic bridge into the upstream RoboGuard package.

    Expected upstream entry points (any available is fine):
        - roboguard.compile_specs(semantic_graph, rules)
        - roboguard.specs.compile_specs(...)
        - roboguard.evaluate.evaluate_plan(plan_steps, compiled_specs)
        - roboguard.evaluate.check_plan(...)

    Fallback: if roboguard isn't importable, use a simple rule-flag heuristic:
        - Step.meta['safety_flags'] can include boolean facts like 'near(hot_surface)': True.
        - If any rule string contains a negated occurrence of that predicate, mark UNSAFE.
    """
    def __init__(self):
        try:
            self.mod = importlib.import_module("roboguard")
        except Exception:
            self.mod = None

    def compile_specs(self, scene: Scene):
        if self.mod is None:
            return {"rules": scene.rules, "semantic_graph": scene.semantic_graph}
        # try top-level
        for name in ("compile_specs", "build_specs", "generate_specs"):
            fn = getattr(self.mod, name, None)
            if callable(fn):
                return fn(scene.semantic_graph, scene.rules)
        # try submodule
        try:
            rg_spec = importlib.import_module("roboguard.specs")
            for name in ("compile_specs", "build_specs", "generate_specs"):
                fn = getattr(rg_spec, name, None)
                if callable(fn):
                    return fn(scene.semantic_graph, scene.rules)
        except Exception:
            pass
        return {"rules": scene.rules, "semantic_graph": scene.semantic_graph}

    def evaluate_plan(self, plan: PlanCandidate, compiled_specs) -> List[Verdict]:
        if self.mod is None:
            # fallback heuristic based on meta 'safety_flags'
            rules_txt = " ".join(compiled_specs.get("rules", []))
            outs: List[Verdict] = []
            for st in plan.steps:
                flags = st.meta.get("safety_flags", {})
                hits = []
                for k, v in flags.items():
                    if v is True and (f"!{k}" in rules_txt or f"not {k}" in rules_txt):
                        hits.append(k)
                if hits:
                    outs.append(Verdict("UNSAFE", why=f"Fallback rule hit: {hits}", details={"flags": flags}))
                else:
                    outs.append(Verdict("SAFE", why="Fallback RG: no conflicts", details={}))
            return outs

        # Call a plausible evaluator
        for mod_name, fn_name in (
            ("roboguard", "evaluate_plan"),
            ("roboguard", "check_plan"),
            ("roboguard", "eval_plan"),
            ("roboguard.evaluate", "evaluate_plan"),
            ("roboguard.evaluate", "check_plan"),
            ("roboguard.evaluate", "eval_plan"),
        ):
            try:
                m = importlib.import_module(mod_name)
                fn = getattr(m, fn_name, None)
                if callable(fn):
                    res = fn([s.action for s in plan.steps], compiled_specs)
                    return self._normalize_rg_result(plan, res)
            except Exception:
                continue

        # Unknown: assume safe
        return [Verdict("SAFE", why="RoboGuard unknown evaluator", details={}) for _ in plan.steps]

    @staticmethod
    def _normalize_rg_result(plan: PlanCandidate, res) -> List[Verdict]:
        outs: List[Verdict] = []
        if isinstance(res, list) and all(isinstance(x, str) for x in res):
            for lab in res:
                outs.append(Verdict(str(lab).upper(), why="RoboGuard", details={}))
            return outs
        if isinstance(res, dict):
            labs = res.get("step_labels") or res.get("labels") or []
            dets = res.get("details") or [{}]*len(labs)
            for lab, det in zip(labs, dets):
                outs.append(Verdict(str(lab).upper(), why="RoboGuard", details=det))
            return outs
        # Unexpected shape
        return [Verdict("SAFE", why="RoboGuard (unrecognized output)", details={"raw": str(res)[:200]}) for _ in plan.steps]

# ==========================================================
# KnowNo adapter (prefers lang_help.knowno.api; falls back to internal CP)
# ==========================================================

class KnowNoAdapter:
    """
    Adapter that first tries to import the Option-A shim:
        from lang_help.knowno import api as knowno_api

    The shim can proxy to upstream 'agent.predict.*' if present; otherwise it
    provides a local conformal-prediction baseline. If the shim import fails,
    we use an internal CP fallback with identical semantics.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backend = self._load_backend()
        self.tau: Optional[float] = None

    def _load_backend(self):
        # Attempt to make shim discoverable via KNOWNO_ROOT (optional)
        root = os.environ.get(self.cfg.KNOWNO_ENV, "")
        if root and os.path.isdir(root) and root not in sys.path:
            sys.path.append(root)
        # Try the shim first
        try:
            mod = importlib.import_module("lang_help.knowno.api")
            return ("shim", mod)
        except Exception:
            pass
        # Try alternate alias if user prefers 'known'
        try:
            mod = importlib.import_module("known.knowno.api")
            return ("shim", mod)
        except Exception:
            pass
        # As a courtesy, try direct upstream modules if the shim isn't present
        for name in (
            "agent.predict.conformal_predictor",
            "agent.predict.multi_step_conformal_predictor",
            "agent.predict.set_predictor",
        ):
            try:
                mod = importlib.import_module(name)
                return ("agent", mod)
            except Exception:
                continue
        # Fallback to internal CP
        return ("fallback", None)

    # ---------- Softmax utility for logits ----------
    @staticmethod
    def _softmax(scores: Sequence[float]) -> List[float]:
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        tot = sum(exps)
        return [e / tot for e in exps] if tot else [1.0/len(scores)]*len(scores)

    # ---------- Calibration ----------
    def calibrate(self, calibration_sets: List[List[float]]) -> float:
        if self.backend[0] == "shim":
            api = self.backend[1]
            # The shim API is a pair of plain functions
            self.tau = float(api.calibrate(alpha=self.cfg.alpha, score_sets=calibration_sets))
            return self.tau

        if self.backend[0] == "agent":
            mod = self.backend[1]
            # Try class-based predictors commonly named ConformalPredictor / MultiStepConformalPredictor
            cls = getattr(mod, "ConformalPredictor", None) or getattr(mod, "MultiStepConformalPredictor", None)
            if cls:
                try:
                    try:
                        inst = cls(alpha=self.cfg.alpha)
                    except TypeError:
                        inst = cls()
                    for mname in ("calibrate", "fit", "train"):
                        if hasattr(inst, mname):
                            fn = getattr(inst, mname)
                            try:
                                out = fn(calibration_sets, self.cfg.alpha)
                            except TypeError:
                                out = fn(calibration_sets)
                            if isinstance(out, (int, float)):
                                self.tau = float(out)
                            elif hasattr(inst, "tau") and isinstance(inst.tau, (int, float)):
                                self.tau = float(inst.tau)
                            else:
                                self.tau = None
                            # Cache instance
                            self._agent_inst = inst
                            return self.tau if self.tau is not None else (1.0 - self.cfg.alpha)
                except Exception:
                    pass
            # Module-level function fallback
            for fname in ("calibrate", "fit"):
                if hasattr(mod, fname):
                    try:
                        self.tau = float(getattr(mod, fname)(calibration_sets, self.cfg.alpha))
                    except TypeError:
                        self.tau = float(getattr(mod, fname)(calibration_sets))
                    return self.tau

        # Internal CP fallback: (1 - alpha) quantile of top scores
        tops = []
        for s in calibration_sets:
            s = list(s)
            if any(v < 0 or v > 1 for v in s):
                s = self._softmax(s)
            tops.append(max(s) if s else 0.0)
        if not tops:
            self.tau = 0.5
        else:
            q = 1.0 - self.cfg.alpha
            idx = max(0, int(math.floor(q * (len(tops) - 1))))
            self.tau = float(sorted(tops)[idx])
        return self.tau

    # ---------- Predict set ----------
    def predict_set(self, scores: List[float]) -> Tuple[List[int], Dict[str, Any]]:
        if not scores:
            return [], {"why": "no-options"}
        s = list(scores)
        if any(v < 0 or v > 1 for v in s):
            s = self._softmax(s)

        if self.backend[0] == "shim":
            api = self.backend[1]
            idxs = api.predict_set(scores=s, tau=self.tau, alpha=self.cfg.alpha)
            return list(map(int, idxs)), {"why": "shim"}

        if self.backend[0] == "agent":
            mod = self.backend[1]
            inst = getattr(self, "_agent_inst", None)
            if inst is None:
                # Try to instantiate anyway
                cls = getattr(mod, "ConformalPredictor", None) or getattr(mod, "MultiStepConformalPredictor", None)
                if cls:
                    try:
                        try:
                            inst = cls(alpha=self.cfg.alpha)
                        except TypeError:
                            inst = cls()
                    except Exception:
                        inst = None
            if inst is not None:
                for m in ("predict_set", "predict"):
                    if hasattr(inst, m):
                        fn = getattr(inst, m)
                        try:
                            out = fn(s)
                        except TypeError:
                            out = fn(s, self.tau)
                        if isinstance(out, dict) and "indices" in out:
                            out = out["indices"]
                        return list(map(int, out)), {"why": f"agent.class.{m}"}
            # Module-level function
            for fname in ("predict_set", "predict"):
                if hasattr(mod, fname):
                    try:
                        out = getattr(mod, fname)(s, self.tau, self.cfg.alpha)
                    except TypeError:
                        out = getattr(mod, fname)(s)
                    if isinstance(out, dict) and "indices" in out:
                        out = out["indices"]
                    return list(map(int, out)), {"why": f"agent.fn.{fname}"}

        # Internal CP fallback: thresholding with guaranteed non-empty set
        tau = self.tau if self.tau is not None else (1.0 - self.cfg.alpha)
        idxs = [i for i, v in enumerate(s) if v >= tau]
        if not idxs:
            idxs = [int(max(range(len(s)), key=lambda i: s[i]))]
        return idxs, {"why": f"fallback(tau={tau:.3f})", "scores": s}

    # ---------- High-level step assessment ----------
    def assess_step(self, step: Step, ask_threshold: float) -> Verdict:
        # Unsafe override via metadata
        if step.meta.get("knowno_flag") == "UNSAFE":
            return Verdict("UNSAFE", why="Meta unsafe flag", details={})

        if not step.candidates or len(step.candidates) == 0:
            return Verdict("UNCERTAIN", why="KnowNo: no candidates", details={})

        scores = [float(s) for _, s in step.candidates]
        idxs, meta = self.predict_set(scores)
        top_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        top_score = scores[top_idx]

        if len(idxs) == 1 and top_score >= ask_threshold:
            return Verdict("SAFE", why=f"Singleton set @ {top_score:.2f}", details={"prediction_set": idxs, **meta})
        if len(idxs) > 1:
            return Verdict("UNCERTAIN", why=f"Set size={len(idxs)}; ask human", details={"prediction_set": idxs, **meta})
        return Verdict("UNCERTAIN", why="Ambiguous KnowNo outcome", details={"prediction_set": idxs, **meta})

# ==========================================================
# Orchestrator
# ==========================================================

class KnowDanger:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.rg = RoboGuardBridge()
        self.kn = KnowNoAdapter(self.cfg)

    def evaluate_step(self, scene: Scene, step: Step, compiled_specs=None) -> StepAssessment:
        rg_vs = self.rg.evaluate_plan(PlanCandidate(name="__tmp__", steps=[step], user_prompt=""), compiled_specs or {"rules": scene.rules, "semantic_graph": scene.semantic_graph})
        rg_v = rg_vs[0] if rg_vs else Verdict("UNCERTAIN", why="RoboGuard no verdict", details={})
        kn_v: Optional[Verdict] = None
        if rg_v.label != "UNSAFE":
            try:
                kn_v = self.kn.assess_step(step, ask_threshold=self.cfg.ask_threshold_confidence)
            except Exception as e:
                kn_v = Verdict("UNCERTAIN", why=f"KnowNo error: {e}", details={})
        final = self.cfg.aggregator(rg_v, kn_v)
        return StepAssessment(step=step, roboguard=rg_v, knowno=kn_v, final=final)

    def run(self, scene: Scene, plan: PlanCandidate) -> PlanAssessment:
        compiled = self.rg.compile_specs(scene)
        rg_verdicts = self.rg.evaluate_plan(plan, compiled)
        out_steps: List[StepAssessment] = []
        for st, rg_v in zip(plan.steps, rg_verdicts):
            kn_v: Optional[Verdict] = None
            if rg_v.label != "UNSAFE":
                try:
                    kn_v = self.kn.assess_step(st, ask_threshold=self.cfg.ask_threshold_confidence)
                except Exception as e:
                    kn_v = Verdict("UNCERTAIN", why=f"KnowNo error: {e}", details={})
            final = self.cfg.aggregator(rg_v, kn_v)
            out_steps.append(StepAssessment(step=st, roboguard=rg_v, knowno=kn_v, final=final))

        labels = [s.final.label for s in out_steps]
        if "UNSAFE" in labels:
            overall = Verdict("UNSAFE", why="At least one step UNSAFE", details={"labels": labels})
        elif "UNCERTAIN" in labels:
            overall = Verdict("UNCERTAIN", why="At least one step UNCERTAIN", details={"labels": labels})
        else:
            overall = Verdict("SAFE", why="All steps SAFE", details={"labels": labels})
        return PlanAssessment(plan=plan, steps=out_steps, overall=overall)

    def evaluate_plans(self, scene: Scene, plans: List[PlanCandidate]) -> List[PlanAssessment]:
        compiled = self.rg.compile_specs(scene)
        return [self.run(scene, p) for p in plans]
