############################
# KnowDanger - Core Module #
############################

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
"""
    
from typing import List, Tuple, Dict, Callable, Optional, TypedDict, Union
import random
import openai
import csv, os, math
from datetime import datetime
import joblib
import numpy as np
import argparse
from enum import Enum
from datetime import datetime

from RoboGuard1.generator import ContextualGrounding
from RoboGuard1.synthesis import ControlSynthesis
from RoboGuard1.roboguard import RoboGuard
from runtime_safety import SafetyContext, NullSensors, make_dynamic_safety_fn

from KnowNo.agent.predict.util import temperature_scaling, get_score

from rg_adapter import RGPlannerValidator


###############
# GPT-4 Logprob Scoring Function
###############

def gpt4_logprob_scoring_fn(prompt: str, model: str = "gpt-4o-mini", top_n: int = 20) -> Dict[str, float]:
    """
    Query OpenAI chat models for top-N token logprobs of the *next* token.
    Requires OPENAI_API_KEY in the environment.
    Returns: { token (str) -> logprob (float) }.
    """
    client = openai.OpenAI()  # uses OPENAI_API_KEY
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=top_n,
        temperature=0, # deterministic for certification
        max_tokens=1,
        top_p=1,
        seed=42,
    )
    # OpenAI returns a list of TopLogprob items for the first generated token:
    # completion.choices[0].logprobs.content[0].top_logprobs -> List[TopLogprob]
    top = completion.choices[0].logprobs.content[0].top_logprobs  # List[TopLogprob]
    # Normalize tokens by stripping spaces; keep the raw logprob values
    token_to_logprob = {}
    for entry in top:
        tok = entry.token
        lp = entry.logprob
        token_to_logprob.setdefault(tok, lp)
        token_to_logprob.setdefault(tok.strip(), lp)  # catch leading-space variants
    return token_to_logprob


###############
# Core Classes
###############

class PlanValidator:
    def __init__(self, forbidden_keywords: List[str]) -> None:
        self.forbidden_keywords = forbidden_keywords

    def validate_plan(self, plan: List[str]) -> List[str]:
        """Removes actions that contain forbidden keywords."""
        safe_plan = []
        for action in plan:
            if any(keyword in action.lower() for keyword in self.forbidden_keywords):
                print(f"[PlanValidator] Removing unsafe action during planning: {action}")
                continue
            safe_plan.append(action)
        return safe_plan

class RoboGuardRewriterWrapper:
    def __init__(self, scene_context: Dict, llm_backend: str = "gpt-4") -> None:
        if not ROBOGUARD_AVAILABLE:
            raise ImportError("RoboGuard is not installed or cannot be imported.")
        self.rewriter = RoboGuardRewriter(llm_backend=llm_backend)
        self.scene_context = scene_context

    def rewrite_plan(self, plan: List[str], prompt: str) -> List[str]:
        try:
            return self.rewriter.rewrite(prompt, plan, self.scene_context)
        except Exception as e:
            print(f"[RoboGuardRewriterWrapper] Rewriting failed: {e}")
            return plan
class FinalStepInterceptor:
    def __init__(self, dynamic_safety_fn: Callable[[str], bool]) -> None:
        self.dynamic_safety_fn = dynamic_safety_fn

    def intercept(self, action: str) -> bool:
        """Checks if an action is safe to execute at the last moment, using a dynamic check."""
        is_safe = self.dynamic_safety_fn(action)
        if not is_safe:
            print(f"[FinalStepInterceptor] Blocking unsafe action at final step: {action}")
        return is_safe

def _canon(token: str) -> str:
    """Map various tokenizations to 'A' or 'B'."""
    t = token.strip().lower()
    # common MC aliases
    if t in {"a", "(a)", "a)", "a.", "yes", "y", "true", "(A)", "Yes", " Yes"}:
        return "A"
    if t in {"b", "(b)", "b)", "b.", "no", "n", "false", "(B)", "No", " No"}:
        return "B"
    return ""  # ignore everything else

def _renorm_ab(lA: float, lB: float, tau: float = 1.0) -> Tuple[float, float]:
    """Return P(A), P(B) after temperature scaling. l* are log-probs."""
    if tau is None or tau <= 0:
        tau = 1.0
    # if a label is missing, treat as very unlikely
    a = 0.0 if lA is None else math.exp(lA / tau)
    b = 0.0 if lB is None else math.exp(lB / tau)
    Z = a + b
    if Z == 0.0:
        return 0.5, 0.5
    return a / Z, b / Z

class LogProbCertifier:
    """
    KnowNo-style runtime certifier:
    - asks the LLM a strict A/B question: "Is it safe? (A) yes (B) no"
    - extracts top-N token logprobs (from scoring_fn)
    - canonizes tokens and renormalizes over {A,B}
    - approves if P(A) >= p_threshold (or gap (P(A) - P(B)) >= gap_threshold)
    """
    def __init__(
        self,
        confidence_threshold: float,
        scoring_fn: Callable[[str], Dict[str, float]],
        log_file: str = "logprob_gaps.csv",
        human_override: bool = True,
        tau: float = 0.7,  # temperature scaling for logprobs
        mode: str = "gap",         # "gap" (pA - pB) or "pA"
        p_threshold: float = 0.80  # used when mode == "pA"
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.scoring_fn = scoring_fn
        self.log_file = log_file
        self.human_override = human_override
        self.tau = tau
        self.mode = mode
        self.p_threshold = p_threshold

        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'prompt', 'pA', 'pB', 'gap'])

    def certify(self, prompt: str) -> bool:
        """
        Decide whether the action is safe, based on P(A) vs P(B).
        Assumes the prompt ends with an A/B instruction like:
          'Is this action safe?\n(A) Yes\n(B) No\nAnswer:'
        """
        # scoring_fn returns a dict: token -> logprob (top-N)
        raw = self.scoring_fn(prompt)

        # fold variants into A/B
        lA, lB = None, None
        for tok, lp in raw.items():
            canon = _canon(tok)
            if canon == "A":
                lA = lp if lA is None else max(lA, lp)  # take best A-like token
            elif canon == "B":
                lB = lp if lB is None else max(lB, lp)

        pA, pB = _renorm_ab(lA, lB, tau=self.tau)
        gap = pA - pB

        # log for later analysis
        with open(self.log_file, mode='a', newline='') as f:
            csv.writer(f).writerow([datetime.now().isoformat(), prompt, f"{pA:.6f}", f"{pB:.6f}", f"{gap:.6f}"])

        print(f"[LogProbCertifier] pA={pA:.3f}, pB={pB:.3f}, gap={gap:.3f} (mode={self.mode})")

        # choose decision rule
        if self.p_threshold is not None:
            confident = pA >= self.p_threshold
        else:
            # interpret confidence_threshold as gap threshold if mode='gap'
            confident = (gap >= self.confidence_threshold) if self.mode == "gap" else (pA >= self.p_threshold)

        print(f"[LogProbCertifier] pA={pA:.3f}, pB={pB:.3f}, gap={gap:.3f}")

        if confident:
            return True
        elif not self.human_override:
            print("[LogProbCertifier] Certifier rejected action due to low confidence.")
            self._log(prompt, pA, pB, gap)
            return False
        else:
            print("[LogProbCertifier] Certifier uncertain, asking human for override.")
            # Ask human for override if enabled
            return self.ask_human(prompt, gap)

    def ask_human(self, prompt: str, gap: float) -> bool:
        print("[Human Check] LLM uncertain (gap = {:.3f}). Prompt:\n{}".format(gap, prompt))
        choice = input("Allow action anyway? (y/n): ").strip().lower()
        return choice == 'y'

    def _log(self, prompt: str, pA: float, pB: float, gap: float) -> None:
        with open(self.log_file, mode='a', newline='') as f:
            csv.writer(f).writerow([datetime.now().isoformat(), prompt, f"{pA:.6f}", f"{pB:.6f}", f"{gap:.6f}"])

class KnowNoBaseline:
    """
    Minimal KnowNo-style baseline:
      • Plan-time keyword filter (optional).
      • Per-step A/B logprob certification (KnowNo-style) with thresholds.
      • No RoboGuard rewriting; No final-step interception.

    Use when you want a lightweight baseline without contextual LTL checks.
    """

    def __init__(self,
                 scoring_fn: Callable[[str], Dict[str, float]],
                 forbidden_keywords: Optional[List[str]] = None,
                 mode: str = "pA",            # or "gap"
                 p_threshold: float = 0.80,   # used if mode == "pA"
                 gap_threshold: float = 0.75, # used if mode == "gap"
                 tau: float = 0.7,
                 human_override: bool = False,
                 log_file: str = "knowno_baseline_gaps.csv") -> None:

        self.plan_validator = PlanValidator(forbidden_keywords or [])

        # Reuse the robust certifier already in knowdanger_core.py
        self.certifier = LogProbCertifier(
            confidence_threshold=gap_threshold,
            scoring_fn=scoring_fn,
            log_file=log_file,
            human_override=human_override,
            tau=tau,
            mode=mode,
            p_threshold=p_threshold,
        )
    # ---- RoboGuard-only baseline (plan-time LTL validation / rewrite) ----
from typing import List, Tuple, Optional

# class RoboGuardBaseline:
#     """
#     RoboGuard-style baseline:
#       • Uses RoboGuard's contextual grounding + control synthesis (if available)
#       • Validates (or repairs) a plan at planning-time
#       • No KnowNo certifier, no final-step interception

#     If RoboGuard isn't importable at runtime, falls back to a simple keyword filter.
#     """

#     def __init__(self, rules: Optional[str] = None, scene_graph: Optional[str] = None):
#         self.rules = rules
#         self.scene_graph = scene_graph
#         self._rg_ok = False
#         self._err = None
#         self._forbidden = ("restricted", "weapon", "harm", "open unknown door", "block exit")

#         try:
#             self._cg = ContextualGrounding()
#             # push scene context
#             if hasattr(self._cg, "update_context"):
#                 self._cg.update_context(self.scene_graph)
#             elif hasattr(self._cg, "set_context"):
#                 self._cg.set_context(self.scene_graph)
#             # optional rules
#             if self.rules and hasattr(self._cg, "update_rules"):
#                 self._cg.update_rules(self.rules)
#             self._synth = ControlSynthesis(self._cg)
#             self._rg_ok = True
#         except Exception as e:
#             self._err = e
#             self._rg_ok = False
#             print(f"[RoboGuardBaseline] Warning: RoboGuard not available ({e}). Using keyword fallback.")

#     def update_context(self, scene_graph: str) -> None:
#         self.scene_graph = scene_graph
#         if self._rg_ok:
#             if hasattr(self._cg, "update_context"):
#                 self._cg.update_context(scene_graph)
#             elif hasattr(self._cg, "set_context"):
#                 self._cg.set_context(scene_graph)

#     def validate_plan(self, plan: List[Tuple[str, str]]) -> Tuple[bool, List[Tuple[Tuple[str, str], bool]]]:
#         """Return (is_safe_plan, [(action, ok_bool), ...])."""
#         if self._rg_ok:
#             if hasattr(self._synth, "validate_action_sequence"):
#                 step_ok = self._synth.validate_action_sequence(plan)
#             else:
#                 step_ok = self._synth.validate_plan(plan)
#             is_safe = all(ok for _, ok in step_ok)
#             return is_safe, step_ok

#         # Fallback: simple forbidden-word filter per action
#         step_ok = []
#         for act in plan:
#             text = f"{act[0]}: {act[1]}".lower()
#             ok = not any(k in text for k in self._forbidden)
#             step_ok.append((act, ok))
#         return all(ok for _, ok in step_ok), step_ok

#     def execute_plan(self, plan: List[Tuple[str, str]]) -> None:
#         """Validate then execute if safe; otherwise print per-step reasons."""
#         is_safe, step_ok = self.validate_plan(plan)
#         if not is_safe:
#             print("[RoboGuardBaseline] Plan rejected:")
#             for a, ok in step_ok:
#                 print("   ", a, "✅" if ok else "❌")
#             return
#         for a, _ in step_ok:
#             print(f"[RoboGuardBaseline] Executing: {a}")


#     @staticmethod
#     def _action_to_text(action) -> str:
#         """Tuple ('primitive','arg') -> 'primitive: arg'; else str(action)."""
#         if isinstance(action, (tuple, list)) and len(action) == 2:
#             return f"{action[0]}: {action[1]}"
#         return str(action)

#     def format_certification_prompt(self, action_text: str) -> str:
#         """Tight A/B prompt so the scorer’s top-N really focuses on {A,B}."""
#         return (
#             f"Action: {action_text}\n"
#             f"Is this action safe?\n"
#             f"(A) Yes\n(B) No\n"
#             f"Answer:"
#         )

#     def execute_plan(self, plan: List) -> None:
#         """
#         Evaluate each step with the A/B certifier and execute only if confident.
#         """
#         candidate_plan = self.plan_validator.validate_plan(plan)

#         for action in candidate_plan:
#             action_text = self._action_to_text(action)
#             prompt = self.format_certification_prompt(action_text)

#             if self.certifier.certify(prompt):
#                 print(f"[KnowNoBaseline] Executing: {action_text}")
#             else:
#                 print(f"[KnowNoBaseline] Blocked (low confidence): {action_text}")

# in knowdanger_core.py

class RoboGuardBaseline:
    """
    RoboGuard-style baseline: planning-time validation only.
    """
    def __init__(self, rules: Optional[str] = None, scene_graph: Optional[str] = None):
        self.scene_graph = scene_graph or ""
        self._rg = RGPlannerValidator(rules=rules)

    def update_context(self, scene_graph: str) -> None:
        self.scene_graph = scene_graph

    def validate_plan(self, plan: List[Tuple[str, str]]):
        step_ok = self._rg.validate(plan, self.scene_graph)
        is_safe = all(ok for _, ok in step_ok)
        return is_safe, step_ok

    def execute_plan(self, plan: List[Tuple[str, str]]) -> None:
        is_safe, step_ok = self.validate_plan(plan)
        if not is_safe:
            print("[RoboGuardBaseline] Plan rejected:")
            for a, ok in step_ok:
                print("   ", a, "✅" if ok else "❌")
            return
        for a, _ in step_ok:
            print(f"[RoboGuardBaseline] Executing: {a}")

def action_to_text(action) -> str:
    # Make a human-readable string from either a tuple or a string.
    if isinstance(action, (tuple, list)) and len(action) == 2:
        return f"{action[0]}: {action[1]}"
    return str(action)
class KnowDanger:
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

    def execute_plan(self, plan: List[Tuple[str, str]]) -> None:
        """
        Filters and executes a validated action plan. If enabled, RoboGuard
        rewrites/validates the plan; then an A/B logprob certifier (or learned
        classifier) checks step-level safety; finally the FinalStepInterceptor
        blocks at execution time if the environment changed.
        """
        # 1) high-level keyword filter
        candidate_plan = self.plan_validator.validate_plan(plan)

        # 2) optional contextual LTL check via RoboGuard
        if self.roboguard:
            scene_graph = self.graph 
            self.roboguard.update_context(scene_graph)
            is_safe, step_results = self.roboguard.validate_plan(candidate_plan)
            if not is_safe:
                print("[KnowDanger] RoboGuard rejected plan:")
                for act, ok in step_results:
                    print("   ", act, "✅" if ok else "❌")
                return
            candidate_plan = [act for act, _ in step_results]

        # 3) step-level certification + final-step interception
        for action in candidate_plan:
            prompt = self.format_certification_prompt(action_to_text(action))

            if self.learned_certifier:
                # Learned certifier path (if you pass one in)
                if not self.learned_certifier(prompt):
                    print(f"[KnowDanger] Refusing (learned certifier): {action}")
                    continue
            elif self.certifier:
                # A/B logprob certifier path
                if not self.certifier.certify(prompt):
                    print(f"[KnowDanger] Refusing (low confidence): {action}")
                    continue

            # 4) dynamic final-step safety
            if self.final_step_interceptor.intercept(action_to_text(action)):
                self.execute_action(action)
            else:
                print(f"[KnowDanger] Execution blocked at final step: {action}")


    def format_certification_prompt(self, action: str) -> str:
        return (
            f"Action: {action}\n"
            f"Is this action safe?\n"
            f"(A) Yes\n(B) No\n"
            f"Answer:"
        )

    @staticmethod
    def execute_action(action: str) -> None:
        """Executes the given action (placeholder for robot control logic)."""
        print(f"[Robot] Executing: {action}")

###############
# CLI Entry Point
###############

class ScorerType(str, Enum):
    GPT4 = "gpt4"
    TOGETHER = "together"

def together_scoring_fn(prompt: str) -> Dict[str, float]:
    openai.api_key = os.getenv("TOGETHER_API_KEY")
    openai.api_base = "https://api.together.xyz/v1"
    try:
        response = openai.Completion.create(
            model="togethercomputer/llama-2-13b-chat",
            prompt=prompt,
            max_tokens=1,
            temperature=0,
            logprobs=5,
            stop=["\n"]
        )
        return response["choices"][0]["logprobs"]["top_logprobs"][0]
    except Exception as e:
        print(f"[Together.ai Error] {e}")
        return {'A': -1.0, 'B': -1.0}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", type=str, choices=["gpt4", "together"], default="gpt4",
                        help="Which LLM to use for logprob scoring: gpt4 or together")
    parser.add_argument("--threshold", type=float, default=1.2, help="Threshold for logprob gap")
    parser.add_argument("--human_override", action="store_true", help="Allow human override on uncertain actions")
    parser.add_argument("--use_roboguard", action="store_true", help="Enable RoboGuard plan rewriting")
    parser.add_argument("--certifier-mode", choices=["pA","gap"], default="pA")
    parser.add_argument("--p-threshold", type=float, default=0.80)
    parser.add_argument("--tau", type=float, default=0.7)

    args = parser.parse_args()

    scorer_fn = gpt4_logprob_scoring_fn if args.scorer == "gpt4" else together_scoring_fn

    plan = [
        "Pick up object A",
        "Move across hallway",
        "Enter restricted area",
        "Deliver package to Room B",
        "Open unknown door"
    ]

    # Define a simple safety policy; TODO: add region labels here.
    safety_ctx = SafetyContext(
        restricted_regions=[
            "restricted_area", "emergency_exit", "fire_escape",
            "road", "construction_zone", "keep_out"
        ],
        min_human_distance_m=1.0,
        keepout_margin_m=0.75,
    )

    # Use NullSensors for now, or swap in a RoboGuardSensors adapter later.
    sensors = NullSensors()

    dynamic_fn = make_dynamic_safety_fn(
        ctx=safety_ctx,
        sensors=sensors,
        strict_unknowns=False,   # set True if you want to fail safe on missing signals
        verbose=True,
    )
    # Initialize the KnowDanger algorithm with the specified parameters
    knowdanger = KnowDanger(
        forbidden_keywords=["restricted", "weapon", "harm"],
        dynamic_safety_fn=dynamic_fn,
        confidence_threshold=args.threshold,   # used if mode="gap"
        scoring_fn=scorer_fn,
        human_override=args.human_override,
        use_roboguard=args.use_roboguard,
        certifier_mode=args.certifier_mode, # "pA" / "gap"
        certifier_tau=args.tau,
        certifier_p_threshold=args.p_threshold, # used if mode="pA"
    )

    knowdanger.execute_plan(plan)
