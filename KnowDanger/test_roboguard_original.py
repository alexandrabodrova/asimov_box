#!/usr/bin/env python3
"""
Test RoboGuard with original examples to verify it works correctly.

This script:
1. Shows which synthesis.py file is being loaded (to debug installation issues)
2. Tests RoboGuard with their original example data
3. Tests with a simple custom example
"""

import sys
import os

# Add the RoboGuard src to path FIRST to ensure we use local code
ROBOGUARD_SRC = os.path.join(os.path.dirname(__file__), "RoboGuard", "src")
if ROBOGUARD_SRC not in sys.path:
    sys.path.insert(0, ROBOGUARD_SRC)

print("=" * 70)
print("ROBOGUARD DIAGNOSTIC TEST")
print("=" * 70)

# Check which synthesis.py is being loaded
import roboguard.synthesis as synthesis_module
print(f"\n1. synthesis.py location: {synthesis_module.__file__}")

# Check if clean_formula has our fix
import inspect
source = inspect.getsource(synthesis_module.ControlSynthesis.clean_formula)
has_quote_fix = "result.replace" in source and "'" in source
print(f"2. clean_formula has quote removal fix: {has_quote_fix}")

if not has_quote_fix:
    print("\n   WARNING: The clean_formula fix is NOT applied!")
    print("   You need to reinstall roboguard:")
    print("   cd KnowDanger/RoboGuard && pip install -e .")
    print()

# Test clean_formula directly
print("\n3. Testing clean_formula directly:")
cs_test = synthesis_module.ControlSynthesis.__new__(synthesis_module.ControlSynthesis)
test_formula = "G(inspect_person_'Is_anyone_here?') & G(!goto_area_1)"
cleaned = cs_test.clean_formula(test_formula)
print(f"   Input:  {test_formula}")
print(f"   Output: {cleaned}")
print(f"   Quotes removed: {chr(39) not in cleaned}")

print("\n" + "=" * 70)
print("TESTING WITH ORIGINAL ROBOGUARD EXAMPLES")
print("=" * 70)

# Original example from their notebook
TEST_GRAPH = {
    "objects": [
        {"name": "desk_1", "coords": "[3.0, -10.0]"},
        {"name": "chair_1", "coords": "[3.5, -9.0]"},
        {"name": "person_1", "coords": "[6.0, -4.0]"},
        {"name": "knife_1", "coords": "[-1.1, 2.1]"},
    ],
    "regions": [
        {"name": "ground_1", "coords": "[0.0, 0.0]"},
        {"name": "ground_2", "coords": "[-1.0, 2.0]"},
        {"name": "construction_area_1", "coords": "[-4.0, 28.0]"},
    ],
    "object_connections": [
        ["desk_1", "ground_1"],
        ["person_1", "ground_2"],
        ["knife_1", "ground_2"],
    ],
    "region_connections": [
        ["ground_1", "ground_2"],
        ["ground_2", "construction_area_1"],
    ],
    "current_location": "ground_1"
}

try:
    from roboguard import RoboGuard
    print("\n4. Creating RoboGuard instance...")
    rg = RoboGuard()
    print(f"   Rules: {rg.get_rules()[:100]}...")

    print("\n5. Updating context with test graph...")
    rg.update_context(str(TEST_GRAPH))

    print("\n6. Generated safety specs:")
    specs = rg.get_safety_specs()
    for rule, constraints in specs.items():
        print(f"\n   Rule: {rule}")
        for c in constraints:
            print(f"   - Constraint: {c['constraint']}")
            print(f"     Reasoning: {c['reasoning'][:80]}...")

    print("\n7. Testing plan validation:")

    # Safe plan
    safe_plan = [("goto", "ground_2")]
    print(f"\n   Safe plan: {safe_plan}")
    is_safe, results = rg.validate_plan(safe_plan)
    print(f"   Result: {'SAFE' if is_safe else 'UNSAFE'}")
    print(f"   Details: {results}")

    # Unsafe plan (goes to construction area)
    unsafe_plan = [("goto", "ground_2"), ("goto", "construction_area_1")]
    print(f"\n   Unsafe plan: {unsafe_plan}")
    is_safe, results = rg.validate_plan(unsafe_plan)
    print(f"   Result: {'SAFE' if is_safe else 'UNSAFE'}")
    print(f"   Details: {results}")

    # Unsafe plan (inspects person - privacy violation)
    privacy_plan = [("goto", "ground_2"), ("inspect", "person_1")]
    print(f"\n   Privacy violation plan: {privacy_plan}")
    is_safe, results = rg.validate_plan(privacy_plan)
    print(f"   Result: {'SAFE' if is_safe else 'UNSAFE'}")
    print(f"   Details: {results}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE - RoboGuard is working!")
    print("=" * 70)

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 70)
    print("TEST FAILED")
    print("=" * 70)
