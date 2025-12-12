#!/usr/bin/env python3
"""
Quick test to verify adapter imports are working correctly in knowdanger_enhanced.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that adapters import correctly"""
    print("Testing adapter imports...\n")

    try:
        from knowdanger.core.knowdanger_enhanced import (
            EnhancedKnowDanger,
            Config,
            ROBOGUARD_AVAILABLE,
            KNOWNO_AVAILABLE,
            INTROPLAN_AVAILABLE
        )
        print("✓ Successfully imported EnhancedKnowDanger\n")

        # Show which adapters loaded
        print("Adapter availability:")
        print(f"  RoboGuard:  {'✓ Available' if ROBOGUARD_AVAILABLE else '✗ Using fallback'}")
        print(f"  KnowNo:     {'✓ Available' if KNOWNO_AVAILABLE else '✗ Using fallback'}")
        print(f"  IntroPlan:  {'✓ Available' if INTROPLAN_AVAILABLE else '✗ Using fallback'}")
        print()

        # Create instance with verbose mode
        print("Creating EnhancedKnowDanger instance with verbose=True:\n")
        config = Config(alpha=0.1, use_introspection=True)
        kd = EnhancedKnowDanger(config, verbose=True)

        print("\n✓ EnhancedKnowDanger created successfully!")
        print(f"  - Config alpha: {kd.cfg.alpha}")
        print(f"  - Introspection enabled: {kd.cfg.use_introspection}")
        print(f"  - Aggregation strategy: {kd.cfg.aggregation_strategy}")

        # Show adapter types
        print("\nAdapter types:")
        print(f"  - RoboGuard: {type(kd.rg).__name__}")
        print(f"  - KnowNo: {type(kd.kn).__name__}")
        print(f"  - IntroPlan: {type(kd.ip).__name__ if kd.ip else 'None'}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
