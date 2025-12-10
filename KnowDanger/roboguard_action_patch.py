"""
RoboGuard Action Filter Patch

RoboGuard's synthesis.py filters out actions not in ROBOT_API (navigation actions).
This causes all manipulation actions (place, lift, carry, etc.) to be filtered out,
resulting in empty plans that always pass validation.

This patch disables the action filtering so RoboGuard can validate manipulation actions.

Usage:
    import roboguard_action_patch
    roboguard_action_patch.patch_roboguard_actions()
"""


def patched_format_plan_for_checking(self, plan):
    """
    Patched version that doesn't filter actions by ROBOT_API

    Original code:
        for i in range(len(plan)):
            if plan[i][0] in ROBOT_API:
                formatted_plan.append(plan[i])

    This filtered out all non-navigation actions, causing empty plans.

    Patched version: Include ALL actions
    """
    # DON'T filter by ROBOT_API - include all actions
    formatted_plan = list(plan)

    # expand plan (adds implied gotos)
    formatted_plan = self.expand_plan(formatted_plan)

    # Format as "action(arg)"
    formatted_plan = [f"{function}({arg})" for function, arg in formatted_plan]

    # get into correct formatting
    for i in range(len(formatted_plan)):
        if formatted_plan[i].startswith("inspect"):
            formatted_plan[i] = (
                formatted_plan[i]
                .split(",")[0]
                .replace("((", "(")
                .replace("'", "")
            )
        # TODO we don't consider answer responses
        if formatted_plan[i].startswith("answer"):
            formatted_plan[i] = "answer"  # don't consider specific response

    # append replan
    formatted_plan = (
        formatted_plan
        if not len(formatted_plan) or formatted_plan[-1].startswith("replan")
        else formatted_plan + ["replan"]
    )

    return formatted_plan


def patch_roboguard_actions():
    """
    Patch RoboGuard's ControlSynthesis to not filter actions by ROBOT_API
    """
    try:
        from roboguard import synthesis

        # Save original for reference
        if not hasattr(synthesis.ControlSynthesis, '_original_format_plan_for_checking'):
            synthesis.ControlSynthesis._original_format_plan_for_checking = (
                synthesis.ControlSynthesis.format_plan_for_checking
            )

        # Replace with patched version
        synthesis.ControlSynthesis.format_plan_for_checking = patched_format_plan_for_checking

        print("✓ Patched RoboGuard to accept all action types (not just ROBOT_API)")
        return True

    except ImportError as e:
        print(f"⚠️  Could not patch RoboGuard actions: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Error patching RoboGuard actions: {e}")
        return False


if __name__ == "__main__":
    print("Testing RoboGuard action patch...")
    print()

    # Test the patch
    success = patch_roboguard_actions()

    if success:
        print()
        print("Patch applied successfully!")
        print()
        print("Now RoboGuard will:")
        print("  ✓ Accept manipulation actions (place, lift, carry, etc.)")
        print("  ✓ Validate them against LLM-generated LTL specs")
        print("  ✓ Not filter them out as 'unknown'")
        print()
        print("Note: The LLM must generate specs that match these actions")
        print("      for the validation to be meaningful.")
    else:
        print("Patch failed - see error above")
