#!/usr/bin/env python3
"""
Diagnose RoboGuard Action Filtering

This script shows what happens when RoboGuard validates plans from our scenes:
1. What actions are in our test plans
2. What actions RoboGuard's ROBOT_API recognizes
3. What gets filtered out during validation
"""

import sys
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "RoboGuard/src"))

print("=" * 80)
print("ROBOGUARD ACTION FILTERING DIAGNOSIS")
print("=" * 80)
print()

# Import RoboGuard's ROBOT_API
try:
    from roboguard.prompts import api as robot_api
    from inspect import getmembers, isfunction

    ROBOT_API = [f[0] for f in getmembers(robot_api, isfunction)]

    print(f"RoboGuard's ROBOT_API ({len(ROBOT_API)} actions):")
    for action in sorted(ROBOT_API):
        print(f"  - {action}")
    print()
except Exception as e:
    print(f"✗ Could not load ROBOT_API: {e}")
    sys.exit(1)

# Load our test scenes
try:
    from knowdanger.core.knowdanger_core import Scene, PlanCandidate

    scene_modules = ["example1_hazard_lab", "example2_breakroom", "example3_photonics"]

    all_actions = set()
    action_counts = {}

    print("Actions in our test scenes:")
    print()

    for scene_name in scene_modules:
        try:
            mod = __import__(f"scenes.{scene_name}", fromlist=["make_scene", "make_plans"])
            scene = mod.make_scene()
            plans = mod.make_plans()

            print(f"{scene_name}:")
            scene_actions = set()

            for plan in plans:
                for step in plan.steps:
                    action = step.action
                    # Action is either "verb(object)" string or (verb, object) tuple
                    if isinstance(action, str):
                        verb = action.split("(")[0] if "(" in action else action
                    elif isinstance(action, tuple):
                        verb = action[0]
                    else:
                        verb = str(action)

                    all_actions.add(verb)
                    scene_actions.add(verb)
                    action_counts[verb] = action_counts.get(verb, 0) + 1

            for action in sorted(scene_actions):
                in_api = "✓" if action in ROBOT_API else "✗"
                print(f"  {in_api} {action}")
            print()

        except Exception as e:
            print(f"  ✗ Failed to load {scene_name}: {e}")
            print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    print(f"Total unique actions in our scenes: {len(all_actions)}")
    print(f"Actions recognized by RoboGuard: {len([a for a in all_actions if a in ROBOT_API])}")
    print(f"Actions filtered out by RoboGuard: {len([a for a in all_actions if a not in ROBOT_API])}")
    print()

    print("Actions that will be FILTERED OUT:")
    filtered = sorted([a for a in all_actions if a not in ROBOT_API])
    if filtered:
        for action in filtered:
            count = action_counts.get(action, 0)
            print(f"  ✗ {action} (used {count} times)")
    else:
        print("  (none)")
    print()

    print("Actions that will be VALIDATED:")
    validated = sorted([a for a in all_actions if a in ROBOT_API])
    if validated:
        for action in validated:
            count = action_counts.get(action, 0)
            print(f"  ✓ {action} (used {count} times)")
    else:
        print("  (none)")
    print()

    if len(filtered) > 0:
        print("⚠️  PROBLEM IDENTIFIED:")
        print(f"   {len(filtered)} out of {len(all_actions)} action types will be filtered out!")
        print("   This means RoboGuard will validate EMPTY plans → everything passes")
        print()
        print("SOLUTIONS:")
        print("  1. Test RoboGuard with navigation-based scenarios (using goto, map_region, etc.)")
        print("  2. Extend ROBOT_API to include manipulation actions")
        print("  3. Modify synthesis.py to not filter unknown actions")
    else:
        print("✓ All actions are recognized by RoboGuard")

except Exception as e:
    print(f"✗ Error loading scenes: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
