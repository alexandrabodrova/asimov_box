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
    
from typing import List, Dict, Callable, Optional, TypedDict, Union
import random
import openai
import os
import csv
from datetime import datetime
import joblib
import numpy as np
import argparse
from enum import Enum

from RoboGuard1.generator import ContextualGrounding
from RoboGuard1.synthesis import ControlSynthesis

###############
# GPT-4 Logprob Scoring Function
###############

def gpt4_logprob_scoring_fn(prompt: str, model: str = "gpt-4o", top_n: int = 20) -> Dict[str, float]:
    """
    Queries OpenAI's GPT-4 model with logprobs enabled to extract the top-N token log probabilities
    for the multiple-choice safety classification task.
    Returns a dictionary mapping token strings (e.g., 'A', 'B') to their logprobs.
    """
    try:
        client = openai.OpenAI()
        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            logprobs=True,
            top_logprobs=top_n,
            seed=42,
            temperature=0,
            max_tokens=1,
        )
        top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        token_to_logprob = {entry.token.strip(): entry.logprob for entry in top_logprobs}
        return token_to_logprob
    except Exception as e:
        print(f"[GPT-4 LogProb Error] {e}")
        return {"A": -1.0, "B": -1.0}


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

class LogProbCertifier:
    def __init__(self, confidence_threshold: float, scoring_fn: Callable[[str], Dict[str, float]], log_file: str = "logprob_gaps.csv", human_override: bool = True) -> None:
        self.confidence_threshold = confidence_threshold
        self.scoring_fn = scoring_fn
        self.log_file = log_file
        self.human_override = human_override

        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'prompt', 'logprob_gap'])

    def certify(self, prompt: str) -> bool:
        """Determines if the action is safe based on log probability gap between top answers."""
        logprobs = self.scoring_fn(prompt)
        sorted_probs = sorted(logprobs.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_probs) < 2:
            return True
        top_token, top_logprob = sorted_probs[0]
        second_token, second_logprob = sorted_probs[1]
        gap = top_logprob - second_logprob
        print(f"[LogProbCertifier] Logprob gap: {gap:.2f} | Top: {top_token}")
        self.log_gap(prompt, gap)

        if top_token == 'A' and gap > self.confidence_threshold:
            return True
        if self.human_override:
            return self.ask_human(prompt, gap)
        return False

    def ask_human(self, prompt: str, gap: float) -> bool:
        """Fallback to human judgment if the model is uncertain."""
        print("[Human Check] LLM uncertain (gap = {:.2f}). Prompt:\n{}".format(gap, prompt))
        choice = input("Allow action anyway? (y/n): ").strip().lower()
        return choice == 'y'

    def log_gap(self, prompt: str, gap: float) -> None:
        """Log the prompt and associated confidence gap to CSV."""
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), prompt, gap])

class KnowDanger:
    def __init__(self, 
                 forbidden_keywords: List[str], 
                 dynamic_safety_fn: Callable[[str], bool],
                 confidence_threshold: Optional[float] = None,
                 scoring_fn: Optional[Callable[[str], Dict[str, float]]] = None,
                 human_override: bool = True,
                 learned_certifier: Optional[Callable[[str, Dict[str, float], Dict], bool]] = None,
                 use_roboguard: bool = True,
                 scene_context: Optional[Dict] = None) -> None:
        self.use_roboguard = use_roboguard and ROBOGUARD_AVAILABLE
        self.scene_context = scene_context or {}
        
        self.plan_validator = PlanValidator(forbidden_keywords)
        self.final_step_interceptor = FinalStepInterceptor(dynamic_safety_fn)
        self.certifier = None
        self.learned_certifier = learned_certifier
        if confidence_threshold is not None and scoring_fn is not None:
            self.certifier = LogProbCertifier(confidence_threshold, scoring_fn, human_override=human_override)
    def execute_plan(self, plan: List[str]) -> None:
        """
        Executes a validated action plan step by step.
        If a learned certifier is available, it is used.
        Otherwise, uses logprob-based certifier to determine safety.
        Each step is intercepted just before execution to allow real-time blocking.
        """
        safe_plan = self.plan_validator.validate_plan(plan)
        for action in safe_plan:
            prompt = self.format_certification_prompt(action)
            logprobs = gpt4_logprob_scoring_fn(prompt)
            if self.learned_certifier:
                metadata = {"scene_complexity": 6, "num_choices": 2}  # placeholder
                if not self.learned_certifier(prompt, logprobs, metadata):
                    print(f"[KnowDanger] Refusing to act due to classifier rejection: {action}")
                    continue
            elif self.certifier:
                if not self.certifier.certify(prompt):
                    print(f"[KnowDanger] Refusing to act due to low confidence in action safety: {action}")
                    continue
            if self.final_step_interceptor.intercept(action):
                self.execute_action(action)
            else:
                print(f"[KnowDanger] Execution stopped for action: {action}")

    def format_certification_prompt(self, action: str) -> str:
        """Formats a yes/no safety prompt to ask the LLM or classifier."""
        return f"Action: {action}\nIs this action safe?\n(A) Yes\n(B) No\nAnswer:"

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

    args = parser.parse_args()

    scorer_fn = gpt4_logprob_scoring_fn if args.scorer == "gpt4" else together_scoring_fn

    plan = [
        "Pick up object A",
        "Move across hallway",
        "Enter restricted area",
        "Deliver package to Room B",
        "Open unknown door"
    ]

    knowdanger = KnowDanger(
        forbidden_keywords=["restricted", "weapon", "harm"],
        dynamic_safety_fn=lambda action: True,  # placeholder
        confidence_threshold=args.threshold,
        scoring_fn=scorer_fn,
        human_override=args.human_override,
        use_roboguard=args.use_roboguard,
        scene_context=scene_context
    )

    knowdanger.execute_plan(plan)
