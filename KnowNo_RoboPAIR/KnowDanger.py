############################
# Asimov Box - Core Module #
############################

"""
Asimov Box: A "guard dog" for LLM-controlled robots.

Key Features:
- Planning-Time Rewriting (inspired by RoboGuard)
- KnowNo-Style Intent Filtering
- Final-Step Interception (new, dynamic safety validation at execution time)
- Epistemic Calibration via LogProbCertifier (KnowNo-style confidence thresholding)
- Optional Human-in-the-Loop override on borderline LLM safety gaps
- Optional Learned Safety Classifier (logprob-aware)

This file defines the Asimov Box classes, safety checkers, and test harness.
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

class AsimovBox:
    def __init__(self, 
                 forbidden_keywords: List[str], 
                 dynamic_safety_fn: Callable[[str], bool],
                 confidence_threshold: Optional[float] = None,
                 scoring_fn: Optional[Callable[[str], Dict[str, float]]] = None,
                 human_override: bool = True,
                 learned_certifier: Optional[Callable[[str, Dict[str, float], Dict], bool]] = None) -> None:
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
            logprobs = together_scoring_fn(prompt)
            if self.learned_certifier:
                metadata = {"scene_complexity": 6, "num_choices": 2}  # placeholder
                if not self.learned_certifier(prompt, logprobs, metadata):
                    print(f"[AsimovBox] Refusing to act due to classifier rejection: {action}")
                    continue
            elif self.certifier:
                if not self.certifier.certify(prompt):
                    print(f"[AsimovBox] Refusing to act due to low confidence in action safety: {action}")
                    continue
            if self.final_step_interceptor.intercept(action):
                self.execute_action(action)
            else:
                print(f"[AsimovBox] Execution stopped for action: {action}")

    def format_certification_prompt(self, action: str) -> str:
        """Formats a yes/no safety prompt to ask the LLM or classifier."""
        return f"Action: {action}\nIs this action safe?\n(A) Yes\n(B) No\nAnswer:"

    @staticmethod
    def execute_action(action: str) -> None:
        """Executes the given action (placeholder for robot control logic)."""
        print(f"[Robot] Executing: {action}")


class KnowNoBaseline:
    def __init__(self, forbidden_keywords: List[str]):
        self.forbidden_keywords = forbidden_keywords

    def execute_plan(self, plan: List[str]) -> None:
        for action in plan:
            if any(keyword in action.lower() for keyword in self.forbidden_keywords):
                print(f"[KnowNoBaseline] Blocking forbidden action: {action}")
                continue
            print(f"[Robot] Executing: {action}")

class RoboGuardBaseline:
    def __init__(self, forbidden_keywords: List[str]):
        self.plan_validator = PlanValidator(forbidden_keywords)

    def execute_plan(self, plan: List[str]) -> None:
        safe_plan = self.plan_validator.validate_plan(plan)
        for action in safe_plan:
            print(f"[Robot] Executing: {action}")

def dynamic_safety_check(action: str) -> bool:
    risky_words = ['cross hallway', 'enter restricted', 'open unknown door']
    for word in risky_words:
        if word in action.lower():
            return random.random() > 0.5
    return True

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
    parser.add_argument("--use_classifier", action="store_true", help="Use learned safety classifier")
    parser.add_argument("--use_threshold", action="store_true", help="Use logprob threshold certifier")
    parser.add_argument("--threshold", type=float, default=1.2, help="Threshold for logprob gap")
    parser.add_argument("--clf_path", type=str, default="safety_classifier.joblib", help="Path to classifier")
    args = parser.parse_args()

    example_plan = [
        "Pick up object A",
        "Move across hallway",
        "Enter restricted area",
        "Deliver package to Room B",
        "Open unknown door"
    ]

    forbidden_keywords = ['restricted', 'weapon', 'harm']

    learned_certifier = None
    threshold_certifier = None

    if args.use_classifier:
        print("\n=== Using Learned Safety Classifier ===")
        learned_certifier = LearnedSafetyCertifier(
            clf_path=args.clf_path,
            feature_extractor=extract_features,
            threshold=0.9
        )

    print("\n=== Launching Asimov Box ===")
    asimov_box = AsimovBox(
        forbidden_keywords=forbidden_keywords,
        dynamic_safety_fn=dynamic_safety_check,
        confidence_threshold=args.threshold if args.use_threshold else None,
        scoring_fn=together_scoring_fn if args.use_threshold else None,
        human_override=True,
        learned_certifier=learned_certifier
    )

    asimov_box.execute_plan(example_plan)

