"""
RoboGuard LTL Syntax Patch

This module patches RoboGuard's ControlSynthesis.clean_formula() method
to convert ALL function calls to atomic propositions, not just ROBOT_API functions.

The original clean_formula only handles navigation functions like goto(), but
our scenes use manipulation actions (place, lift, carry) and predicates (near,
mass, temp) which need to be converted to valid spot LTL syntax.

Usage:
    import roboguard_ltl_patch
    roboguard_ltl_patch.patch_roboguard_synthesis()
"""

import re


def enhanced_clean_formula(self, ltl_formula: str) -> str:
    """
    Enhanced version of clean_formula that converts ALL function calls
    to atomic propositions for spot LTL, while preserving LTL operators.

    Converts: function(arg1, arg2) -> function_arg1_arg2

    Preserves LTL operators: G, F, X, U, R, W, M (temporal)
                             !, &, |, ->, <-> (logical)

    This handles:
    - Actions: place(x, y), lift(x), carry(x)
    - Predicates: near(x, y), mass(x), temp(x)
    - Nested: near(bench1, hotplate | flame) -> near_bench1_hotplate_OR_flame
    """
    # LTL temporal operators that should NOT be converted
    LTL_OPERATORS = {'G', 'F', 'X', 'U', 'R', 'W', 'M'}

    formula = ltl_formula

    def replace_func_call(match):
        func_name = match.group(1)
        args = match.group(2)

        # Don't convert LTL temporal operators
        if func_name in LTL_OPERATORS:
            return match.group(0)  # Return unchanged

        # Clean the arguments
        # Replace | with _OR_
        clean_args = args.replace(" | ", "_OR_").replace("|", "_OR_")
        # Replace commas
        clean_args = clean_args.replace(", ", "_").replace(",", "_")
        # Replace comparison operators
        clean_args = clean_args.replace(" < ", "_LT_").replace(" > ", "_GT_")
        clean_args = clean_args.replace(" <= ", "_LEQ_").replace(" >= ", "_GEQ_")
        clean_args = clean_args.replace(" == ", "_EQ_").replace(" != ", "_NEQ_")
        clean_args = clean_args.replace("<", "_LT_").replace(">", "_GT_")
        # Replace arithmetic - and +
        # But be careful with negative numbers and operators
        clean_args = re.sub(r'\s*-\s*', "_MINUS_", clean_args)
        clean_args = re.sub(r'\s*\+\s*', "_PLUS_", clean_args)
        clean_args = re.sub(r'\s*\*\s*', "_TIMES_", clean_args)
        clean_args = re.sub(r'\s*/\s*', "_DIV_", clean_args)
        # Remove any remaining spaces
        clean_args = clean_args.replace(" ", "_")

        return f"{func_name}_{clean_args}"

    # Strategy: Process innermost parentheses first
    # Match function calls without nested parens: func(args without parens)
    # Repeat until no more simple function calls remain

    max_iterations = 20  # Increased for deeply nested formulas
    for i in range(max_iterations):
        # Match func(args) where args doesn't contain parentheses
        # This processes innermost calls first
        pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\(([^()]+)\)"
        new_formula = re.sub(pattern, replace_func_call, formula)

        if new_formula == formula:
            # No more changes
            break
        formula = new_formula
    else:
        print(f"⚠️  Warning: clean_formula may not have fully processed formula after {max_iterations} iterations")
        print(f"   Remaining formula: {formula[:200]}...")

    return formula


def patch_roboguard_synthesis():
    """
    Patch RoboGuard's ControlSynthesis class to use enhanced_clean_formula
    """
    try:
        from roboguard import synthesis

        # Save original for reference
        if not hasattr(synthesis.ControlSynthesis, '_original_clean_formula'):
            synthesis.ControlSynthesis._original_clean_formula = synthesis.ControlSynthesis.clean_formula

        # Replace with enhanced version
        synthesis.ControlSynthesis.clean_formula = enhanced_clean_formula

        print("✓ Patched RoboGuard ControlSynthesis.clean_formula for full LTL support")
        return True

    except ImportError as e:
        print(f"⚠️  Could not patch RoboGuard synthesis: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Error patching RoboGuard synthesis: {e}")
        return False


if __name__ == "__main__":
    # Test the enhanced cleaning
    print("Testing enhanced LTL cleaning...")
    print()

    test_cases = [
        "place(solvent_bottle, bench1)",
        "near(bench1, hotplate | flame)",
        "mass(solvent_bottle) < payload_limit",
        "pour(A,B)",
        "G(place(x, y) -> !near(y, hotplate))",
        "microwave(foil_box, t, p)",
        "temp(batt1) < T_safe",
    ]

    # Create a mock self object
    class MockSelf:
        pass

    mock_self = MockSelf()

    for test in test_cases:
        result = enhanced_clean_formula(mock_self, test)
        print(f"Input:  {test}")
        print(f"Output: {result}")
        print()
