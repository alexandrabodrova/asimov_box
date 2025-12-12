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

    # Second pass: Handle comparison and arithmetic expressions that remain
    # These appear after function calls are converted, like: mass_solvent_bottle < payload_limit - margin
    # We need to convert these to single atomic propositions for spot

    def clean_atomic_expr(match):
        """Convert expressions with comparisons/arithmetic to single atomic prop"""
        expr = match.group(0)

        # Replace comparison operators first (with spaces)
        cleaned = expr.replace(' <= ', '_LEQ_').replace(' >= ', '_GEQ_')
        cleaned = cleaned.replace(' < ', '_LT_').replace(' > ', '_GT_')
        cleaned = cleaned.replace(' == ', '_EQ_').replace(' != ', '_NEQ_')

        # Replace comparison operators without spaces
        cleaned = cleaned.replace('<=', '_LEQ_').replace('>=', '_GEQ_')
        cleaned = cleaned.replace('<', '_LT_').replace('>', '_GT_')
        cleaned = cleaned.replace('==', '_EQ_').replace('!=', '_NEQ_')

        # Replace arithmetic operators (but NOT -> which is LTL implication)
        # Only replace - when it's arithmetic (surrounded by identifiers/numbers)
        cleaned = re.sub(r'(\w)\s*-\s*(\w)', r'\1_MINUS_\2', cleaned)
        cleaned = re.sub(r'(\w)\s*\+\s*(\w)', r'\1_PLUS_\2', cleaned)
        cleaned = re.sub(r'(\w)\s*\*\s*(\w)', r'\1_TIMES_\2', cleaned)
        cleaned = re.sub(r'(\w)\s*/\s*(\w)', r'\1_DIV_\2', cleaned)

        # Remove remaining spaces
        cleaned = cleaned.replace(' ', '_')

        return cleaned

    # Match sequences of identifiers, numbers, and operators (but not LTL operators)
    # This captures things like: mass_solvent_bottle < payload_limit - margin
    # Pattern: identifier followed by one or more (comparison/arithmetic operator + identifier/number)
    # BUT stop at LTL operators: &, |, ->, <->, (, ), ;
    # Use negative lookahead to avoid matching ->
    pattern = r'[a-zA-Z_][a-zA-Z0-9_]*(?:\s*(?:<=|>=|==|!=|<|>|(?<![->])-(?!>)|\+|\*|/)\s*[a-zA-Z0-9_]+)+'
    formula = re.sub(pattern, clean_atomic_expr, formula)

    # Third pass: Handle invalid sequence operators
    # LLM sometimes generates (expr1; expr2) which is invalid in spot when inside G()
    # Convert these to conjunction: (expr1; expr2) -> (expr1 & expr2)
    # This loses temporal ordering but makes the formula valid
    def replace_sequence(match):
        """Replace (a; b) with (a & b) to make it valid LTL"""
        # Get content between parentheses
        content = match.group(1)
        # Replace ; with &
        fixed = content.replace(';', ' &')
        return f'({fixed})'

    # Match (expr; expr) patterns
    # This handles cases like: (pour_A_beaker_B; pour_C_beaker_B)
    pattern = r'\(([^()]+;[^()]+)\)'
    formula = re.sub(pattern, replace_sequence, formula)

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
