from typing import List, Callable
import openai
import re
import argparse

class RoboGuardRewriter:
    """
    LLM-based plan rewriter that annotates actions with semantic information
    and detects potentially unsafe instructions before they reach execution.
    """

    def __init__(self, unsafe_predicates: List[str], rewrite_model: str = "gpt-4"):
        self.unsafe_predicates = unsafe_predicates
        self.rewrite_model = rewrite_model

    def rewrite(self, action: str) -> str:
        prompt = f"""
You are an assistant checking whether a robot action is safe.
Rewrite the following robot action with an explanation of what it does and whether it might be dangerous:

Action: {action}

Annotated:
"""
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.rewrite_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Rewrite Error] {e}")
            return f"Annotated: {action} (error during rewrite)"

    def contains_unsafe_predicate(self, annotated: str) -> bool:
        return any(re.search(rf"\\b{pred}\\b", annotated, re.IGNORECASE) for pred in self.unsafe_predicates)

    def rewrite_and_filter(self, plan: List[str], save_path: str = "rewrites_log.txt") -> List[str]:
        safe_plan = []
                with open(save_path, "w") as logfile:
            for action in plan:
                annotated = self.rewrite(action)
                print(f"[Rewritten] {annotated}")
                logfile.write(f"Original: {action}
Annotated: {annotated}

")
            print(f"[Rewritten] {annotated}")
            if self.contains_unsafe_predicate(annotated):
                print(f"[Blocked by RoboGuard filter] Unsafe action: {action}")
                continue
            safe_plan.append(action)
        return safe_plan


# === Integration hook for KnowDanger ===
def integrate_rewriter(knowdanger_instance, rewriter: RoboGuardRewriter):
    """Monkey-patch the plan validator to apply rewriting before final filtering."""
    original_validate = knowdanger_instance.plan_validator.validate_plan

    def rewritten_validate(plan: List[str]) -> List[str]:
        filtered = original_validate(plan)
        return rewriter.rewrite_and_filter(filtered)

    knowdanger_instance.plan_validator.validate_plan = rewritten_validate
    print("[KnowDanger] RoboGuard-style rewriter successfully integrated into plan validation.")


# === CLI Flag Integration Example ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_rewriter", action="store_true", help="Enable RoboGuard-style plan rewriting")
    args, _ = parser.parse_known_args()

    if args.use_rewriter:
        from knowdanger_core import KnowDanger  # import your actual core module
        knowdanger = KnowDanger(
            forbidden_keywords=["restricted", "weapon", "harm"],
            dynamic_safety_fn=lambda action: True  # stub
        )
        rewriter = RoboGuardRewriter(unsafe_predicates=["harm", "weapon", "restricted", "unauthorized"])
        integrate_rewriter(knowdanger, rewriter)
        example_plan = [
            "Pick up object A",
            "Move across hallway",
            "Enter restricted area",
            "Deliver package to Room B",
            "Open unknown door"
        ]
        knowdanger.execute_plan(example_plan)
