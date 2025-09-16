from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import math

from knowdanger.adapters.roboguard_adapter import RoboGuardAdapter, action_to_text, plan_to_strings, Action

RULES_DEFAULT: List[str] = [
    "Never collide with a person.",
    "Never approach a person at high speed.",
    "Never block an emergency exit.",
    "Never enter a restricted area.",
    "Never handle explosives or weapons.",
]

HARMFUL_TOKENS = [
    "collide(", "detonate(", "handle(explosive", "weapon", "block(exit", "restricted_area",
    "near_person", "towards_person"
]

@dataclass
class Certifier:
    mode: str = "pA"
    p_threshold: float = 0.80
    tau: float = 0.7
    def score(self, a_text: str) -> float:
        has_harm = any(tok in a_text for tok in HARMFUL_TOKENS)
        logit = -3.0 if has_harm else 3.0
        if self.mode == "gap":
            return (logit - (-logit)) / max(self.tau, 1e-3)
        return 1.0 / (1.0 + math.exp(-(logit) / max(self.tau, 1e-3)))
    def certify(self, a_text: str) -> bool:
        pA = self.score(a_text)
        return (pA >= self.p_threshold) if self.mode != "gap" else (pA >= 0.0)

class FinalStepInterceptor:
    def intercept(self, action_text: str) -> bool:
        return True

class PlanValidator:
    def __init__(self, harmful_keywords: Optional[Sequence[str]] = None) -> None:
        self.harmful = list(harmful_keywords) if harmful_keywords is not None else HARMFUL_TOKENS
    def validate_plan(self, plan: Sequence[Action]) -> List[Action]:
        return list(plan)

class KnowDanger:
    def __init__(
        self,
        use_roboguard: bool = True,
        certifier_mode: str = "pA",
        p_threshold: float = 0.80,
        tau: float = 0.7,
        rules: Optional[Sequence[str]] = None,
    ) -> None:
        self.use_roboguard = use_roboguard
        self.plan_validator = PlanValidator()
        self.certifier = Certifier(mode=certifier_mode, p_threshold=p_threshold, tau=tau)
        self.final_step_interceptor = FinalStepInterceptor()
        self._rg: Optional[RoboGuardAdapter] = RoboGuardAdapter(rules=rules or RULES_DEFAULT) if use_roboguard else None

    def fit_guard(self, scene_graph: Dict[str, Any], rules: Optional[Sequence[str]] = None) -> None:
        if self._rg is None:
            return
        n = self._rg.fit(scene_graph, rules or RULES_DEFAULT)
        status = "compiled" if n > 0 else "vacuous"
        print(f"[KnowDanger] RoboGuard specs_status: {status} (count={n})")
        for line in self._rg.diagnostics:
            if "WARNING" in line or "roboguard." in line or "set_" in line:
                print(line)

    def execute_plan(self, plan: Sequence[Action], *, verbose: bool = False) -> Tuple[bool, List[Tuple[str, bool]]]:
        plan_strs = plan_to_strings(plan)
        if self._rg is not None:
            ok_rg, step_rg = self._rg.check_plan(plan_strs)
            if verbose:
                if not ok_rg:
                    print("[KnowDanger] RoboGuard rejected plan:")
                    for a, ok in step_rg:
                        print(f"   {a} {'✅' if ok else '❌'}")
                else:
                    print("[KnowDanger] RoboGuard accepted plan.")
            if not ok_rg:
                return False, step_rg
        candidate = self.plan_validator.validate_plan(plan_strs)
        final_steps: List[Tuple[str, bool]] = []
        for a in candidate:
            a_txt = action_to_text(a)
            if not self.certifier.certify(a_txt):
                if verbose: print(f"[KnowDanger] Refusing due to low confidence: {a_txt}")
                final_steps.append((a_txt, False)); continue
            ok = self.final_step_interceptor.intercept(a_txt)
            final_steps.append((a_txt, bool(ok)))
        return all(ok for _, ok in final_steps), final_steps
