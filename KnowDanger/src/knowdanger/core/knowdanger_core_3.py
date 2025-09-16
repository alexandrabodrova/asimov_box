from typing import List, Tuple, Dict, Callable, Optional, Union
import math, csv, os
from datetime import datetime
import openai
import inspect

# ---- Imports you already had / local helpers ----
from RoboGuard1.roboguard import RoboGuard  # use the real RoboGuard class

Action = Union[str, Tuple[str, str], List[str]]  # string or (verb,arg)

# --------------------------------------
# Basic Roboguard Natural Language rules
# --------------------------------------
RULES = [
  "Never collide with a person.",
  "Never approach a person at high speed.",
  "Never handle or transport explosives.",
  "Never detonate any explosive device.",
  "Never block exits or egress routes.",
  "Never enter a restricted area.",
]

# -----------------------
# Utility: action -> text
# -----------------------
def action_to_text(action: Action) -> str:
    if isinstance(action, str):
        return action
    if isinstance(action, (tuple, list)) and len(action) == 2:
        return f"{action[0]}: {action[1]}"
    return str(action)

def _to_rg_str(action: Action) -> str:
    # RoboGuard expects strings like verb(arg)
    if isinstance(action, (tuple, list)) and len(action) == 2:
        v, a = str(action[0]).strip(), str(action[1]).strip()
        return f"{v}({a})"
    if isinstance(action, str):
        # allow "verb: arg" — convert to verb(arg)
        s = action.strip()
        if ":" in s:
            v, a = s.split(":", 1)
            return f"{v.strip()}({a.strip()})"
        # already a simple verb? accept as verb()
        if "(" in s and s.endswith(")"):
            return s
        return f"{s}()"  # no-argument case
    return f"{str(action)}()"

# ---------------------
# Simple plan validator
# ---------------------
class PlanValidator:
    def __init__(self, forbidden_keywords: List[str]) -> None:
        self.forbidden_keywords = [k.lower() for k in (forbidden_keywords or [])]

    def validate_plan(self, plan: List[Action]) -> List[Action]:
        safe: List[Action] = []
        for a in plan:
            if any(k in action_to_text(a).lower() for k in self.forbidden_keywords):
                print(f"[PlanValidator] Removing unsafe action during planning: {a}")
                continue
            safe.append(a)
        return safe
    
    # # Backward-compatible alias expected by callers
    # def validate_plan(self, plan: List[Action]) -> List[Action]:
    #     return self.validate_plan1(plan)

# -------------------------
# Final-step interceptor API
# -------------------------
class FinalStepInterceptor:
    def __init__(self, dynamic_safety_fn: Callable[[str], bool]) -> None:
        self.dynamic_safety_fn = dynamic_safety_fn

    def intercept(self, action: Action) -> bool:
        txt = action_to_text(action)
        ok = self.dynamic_safety_fn(txt)
        if not ok:
            print(f"[FinalStepInterceptor] Blocking unsafe action at final step: {txt}")
        return ok

# ---------------------------------------
# Log-prob based KnowNo-style certifier
# ---------------------------------------
def _canon(tok: str) -> str:
    t = tok.strip().lower()
    if t in {"a", "(a)", "a)", "a.", "yes", "y", "true"}:
        return "A"
    if t in {"b", "(b)", "b)", "b.", "no", "n", "false"}:
        return "B"
    return ""

def _renorm_ab(lA: Optional[float], lB: Optional[float], tau: float = 1.0) -> Tuple[float, float]:
    if not tau or tau <= 0:
        tau = 1.0
    a = 0.0 if lA is None else math.exp(lA / tau)
    b = 0.0 if lB is None else math.exp(lB / tau)
    Z = a + b
    return (0.5, 0.5) if Z == 0 else (a / Z, b / Z)

class LogProbCertifier:
    def __init__(self,
                 confidence_threshold: float,
                 scoring_fn: Callable[[str], Dict[str, float]],
                 log_file: str = "logprob_gaps.csv",
                 human_override: bool = True,
                 tau: float = 0.7,
                 mode: str = "pA",
                 p_threshold: float = 0.80) -> None:
        self.confidence_threshold = confidence_threshold
        self.scoring_fn = scoring_fn
        self.log_file = log_file
        self.human_override = human_override
        self.tau = tau
        self.mode = mode
        self.p_threshold = p_threshold

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "prompt", "pA", "pB", "gap"])

    def certify(self, prompt: str) -> bool:
        raw = self.scoring_fn(prompt)
        lA, lB = None, None
        for tok, lp in raw.items():
            c = _canon(tok)
            if c == "A": lA = lp if lA is None else max(lA, lp)
            if c == "B": lB = lp if lB is None else max(lB, lp)
        pA, pB = _renorm_ab(lA, lB, tau=self.tau)
        gap = pA - pB

        with open(self.log_file, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().isoformat(), prompt, f"{pA:.6f}", f"{pB:.6f}", f"{gap:.6f}"])

        print(f"[LogProbCertifier] pA={pA:.3f}, pB={pB:.3f}, gap={gap:.3f} (mode={self.mode})")

        if self.mode == "pA":
            confident = pA >= self.p_threshold
        else:
            confident = gap >= self.confidence_threshold

        if confident:
            return True
        if not self.human_override:
            print("[LogProbCertifier] Certifier rejected action due to low confidence.")
            return False
        # human-in-the-loop fallback
        choice = input("Allow action anyway? (y/n): ").strip().lower()
        return choice == "y"

# --------------- RoboGuard Baseline (use upstream class directly) ---------------
class RoboGuardBaseline:
    """
    RoboGuard baseline using the upstream RoboGuard API:
    """
    def __init__(self, scene_graph: dict, rules: list[str]): # : dict | None = None, rules: list[str] | None = None
        self._rg = RoboGuard()
        self.scene_graph = None
        self.rules = rules or []
        self._specs = None
        if scene_graph is not None:
            self.set_scene_graph(scene_graph)

    def set_scene_graph(self, graph: dict):
        self.scene_graph = graph
        # 1) Give context to RG
        if hasattr(self._rg, "update_context"):
            self._rg.update_context(graph)
        # 2) Synthesize specs from rules (+CoT) if the API exposes it
        self._specs = None
        for m in ("generate_specs", "contextualize_rules", "build_specs"):
            if hasattr(self._rg, m):
                fn = getattr(self._rg, m)
                try:
                    # try common argument shapes
                    try:
                        self._specs = fn(self.rules, chain_of_thought=True)
                    except TypeError:
                        self._specs = fn(self.rules)
                except Exception as e:
                    print(f"[RoboGuardBaseline] Spec synthesis via {m} failed: {e}")
                break
        if self._specs is None:
            print("[RoboGuardBaseline] WARNING: no specs synthesized; guard will be vacuous.")

    def _to_rg_strings(self, plan):
        out = []
        for a in plan:
            if isinstance(a, (tuple, list)) and len(a) == 2:
                v, x = str(a[0]).strip(), str(a[1]).strip()
                out.append(f"{v}({x})")
            elif isinstance(a, str):
                s = a.strip()
                if ":" in s:
                    v, x = s.split(":", 1)
                    out.append(f"{v.strip()}({x.strip()})")
                else:
                    out.append(s if "(" in s and s.endswith(")") else f"{s}()")
            else:
                out.append(str(a))
        return out

    def validate_plan(self, plan):
        plan_strs = self._to_rg_strings(plan)
        # Prefer signatures that accept specs; otherwise rely on internal state.
        sig = inspect.signature(self._rg.validate_plan)
        try:
            if len(sig.parameters) >= 2 and self._specs is not None:
                raw = self._rg.validate_plan(plan_strs, self._specs)
            else:
                raw = self._rg.validate_plan(plan_strs)
        except Exception as e:
            raise RuntimeError(f"RoboGuard.validate_plan failed: {e}")

        # Normalize returns: (bool, details) | list[(a,bool)] | list[bool] | bool
        step_ok = []
        if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], bool):
            is_safe, details = raw
            if (isinstance(details, list) and details and isinstance(details[0], tuple)
                and len(details[0]) >= 2 and isinstance(details[0][1], bool)):
                step_ok = [(str(a), bool(ok)) for (a, ok, *_) in details]
            else:
                step_ok = [(s, bool(is_safe)) for s in plan_strs]
        elif isinstance(raw, list) and raw and isinstance(raw[0], tuple) and isinstance(raw[0][1], bool):
            step_ok = [(str(a), bool(ok)) for (a, ok) in raw]
        elif isinstance(raw, list) and all(isinstance(x, bool) for x in raw):
            step_ok = list(zip(plan_strs, raw))
        elif isinstance(raw, bool):
            step_ok = [(p, raw) for p in plan_strs]
        else:
            print("[RoboGuardBaseline] Unexpected validate_plan return; marking all steps as UNSAFE.")
            step_ok = [(p, False) for p in plan_strs]   # ← fail-safe deny

        return all(ok for _, ok in step_ok), step_ok


# ----------------
# KnowDanger core
# ----------------
class KnowDanger:
    """
    KnowDanger pipeline with optional RoboGuard plan-time validation, KnowNo-style
    certification, and final-step interception. Uses the *real* RoboGuard API.
    """

    def __init__(self,
                 forbidden_keywords,
                 dynamic_safety_fn,
                 confidence_threshold: float = None,
                 scoring_fn: Callable[[str], Dict[str, float]] = None,
                 human_override: bool = True,
                 learned_certifier=None,
                 use_roboguard: bool = True,
                 graph=None,
                 certifier_mode: str = "pA",
                 certifier_tau: float = 0.7,
                 certifier_p_threshold: float = 0.80):
        self.roboguard = RoboGuard() if use_roboguard else None
        self._rg_graph_ready = False
        self.scene_graph = None  # textual scene graph
        self.graph = graph or {}
        self.plan_validator = PlanValidator(forbidden_keywords)
        self.final_step_interceptor = FinalStepInterceptor(dynamic_safety_fn)
        self.certifier = None
        self.learned_certifier = learned_certifier

        if confidence_threshold is not None and scoring_fn is not None:
            self.certifier = LogProbCertifier(
                confidence_threshold=confidence_threshold,
                scoring_fn=scoring_fn,
                human_override=human_override,
                tau=certifier_tau,
                mode=certifier_mode,
                p_threshold=certifier_p_threshold,
            )

    # NEW: let caller push scene graph text once
    def set_scene_graph(self, scene_graph: str) -> None:
        self.scene_graph = str(scene_graph)
        if self.roboguard is not None:
            self.roboguard.update_context(self.scene_graph)
            self._rg_graph_ready = True

    def _format_certification_prompt(self, action: Action) -> str:
        t = action_to_text(action)
        return f"Action: {t}\nIs this action safe?\n(A) Yes\n(B) No\nAnswer:"

    def execute_action(self, action_text: str) -> None:
        print(f"[Robot] Executing: {action_text}")

    def execute_plan(self, plan: List[Action]) -> None:
        """
        Plan-time RG validation (if enabled) → KnowNo-style certification → final-step check.
        """
        candidate_plan: List[Action] = plan

        # --- RoboGuard plan-time validation (exact usage) ---
        if self.roboguard is not None:
            if not self._rg_graph_ready and self.scene_graph:
                self.roboguard.update_context(self.scene_graph)
                self._rg_graph_ready = True

            plan_strs = [_to_rg_str(a) for a in candidate_plan]
            try:
                raw = self.roboguard.validate_plan(plan_strs)  # ONE arg only
            except AssertionError as e:
                # Likely: "must initialize synthesizer via `update_context`"
                print(f"[KnowDanger] RoboGuard assertion: {e}")
                return

            step_ok: List[Tuple[str, bool]] = []
            # Handle upstream RoboGuard signature: (is_safe: bool, details: list|dict)
            if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], bool):
                is_safe, details = raw
                if (isinstance(details, list) and details and isinstance(details[0], tuple)
                    and len(details[0]) >= 2 and isinstance(details[0][1], bool)):
                    step_ok = [(str(a), bool(ok)) for (a, ok, *_) in details]
                else:
                    step_ok = [(s, bool(is_safe)) for s in plan_strs]
            elif isinstance(raw, list):
                ...
            elif isinstance(raw, bool):
                ...
            else:
                step_ok = [(s, False) for s in plan_strs]


            is_safe = all(ok for _, ok in step_ok)
            if not is_safe:
                print("[KnowDanger] RoboGuard rejected plan:")
                for a, ok in step_ok:
                    print("  ", a, "✅" if ok else "❌")
                return

            # keep only allowed steps (if RG filtered) and convert back to Action strings
            allowed_strs = [a for a, ok in step_ok if ok]
            candidate_plan = allowed_strs  # these are already strings in verb(arg) format

        # --- Keyword filter baseline (kept as before) ---
        candidate_plan = self.plan_validator.validate_plan(candidate_plan)

        # --- Step-level certification + final-step interception ---
        for action in candidate_plan:
            prompt = self._format_certification_prompt(action)
            if self.certifier and not self.certifier.certify(prompt):
                print(f"[KnowDanger] Refusing due to low confidence: {action_to_text(action)}")
                continue
            if self.final_step_interceptor.intercept(action):
                self.execute_action(action_to_text(action))
            else:
                print(f"[KnowDanger] Execution blocked at final step: {action}")
