# asimov_box/KnowDanger/runtime_safety.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Protocol, Callable, List
import math
import re
import time

# ---------------------------
# Sensor adapter interfaces
# ---------------------------

class SensorAdapter(Protocol):
    """Abstracts how we query the world at execution time."""
    def current_pose(self) -> Optional[Tuple[float, float, float]]:
        """(x,y,heading) in meters/radians if available; else None."""
        ...
    def nearest_human_distance(self) -> Optional[float]:
        """Meters to nearest detected human; None if unknown/no detection."""
        ...
    def region_of(self, pos_or_label) -> Optional[str]:
        """Map a position or a name to a semantic region label, if available."""
        ...
    def path_intersects_restricted(self, goal_xy: Tuple[float, float], restricted: Iterable[str], margin_m: float) -> bool:
        """True if planned path to goal intersects any restricted regions (with margin)."""
        ...

class NullSensors:
    """Fallback sensor adapter if you don't have live sensors yet."""
    def current_pose(self) -> Optional[Tuple[float, float, float]]:
        return None
    def nearest_human_distance(self) -> Optional[float]:
        return None
    def region_of(self, pos_or_label) -> Optional[str]:
        return None
    def path_intersects_restricted(self, goal_xy, restricted, margin_m) -> bool:
        return False

# Example adapter sketch if you wire RoboGuard/world model later.
class RoboGuardSensors(NullSensors):
    """Illustrative stub showing where to call into RoboGuard's world model."""
    def __init__(self, roboguard_obj):
        self.rg = roboguard_obj
    # Implement methods by delegating to rg's map/graph APIs when available.

# ---------------------------
# Safety policy config
# ---------------------------

@dataclass
class SafetyContext:
    """Policy thresholds and semantics for final-step checks."""
    restricted_regions: List[str]
    min_human_distance_m: float = 0.8
    keepout_margin_m: float = 0.5
    forbid_verbs: Tuple[str, ...] = (
        "enter restricted", "cross hallway", "open unknown door",
        "follow someone", "surveil", "record secretly", "detonate",
        "block exit", "block road",
    )
    # Words that suggest high-speed/aggressive motion
    speed_risky_adverbs: Tuple[str, ...] = ("quickly", "rapidly", "fast", "asap")

# ---------------------------
# Utilities
# ---------------------------

_coord_pat = re.compile(r"(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)")  # matches "x,y"

def _parse_goal_xy(action_text: str) -> Optional[Tuple[float, float]]:
    """Very light parser for goals formatted like 'move to x,y'."""
    m = _coord_pat.search(action_text)
    if not m:
        return None
    try:
        return (float(m.group(1)), float(m.group(3)))
    except Exception:
        return None

def _contains_any(text: str, needles: Iterable[str]) -> bool:
    t = text.lower()
    return any(n in t for n in needles)

# ---------------------------
# Factory for the dynamic safety function
# ---------------------------

def make_dynamic_safety_fn(
    ctx: SafetyContext,
    sensors: SensorAdapter,
    strict_unknowns: bool = False,
    verbose: bool = True,
) -> Callable[[str], bool]:
    """
    Returns a function: action_text -> bool (True if safe to execute now).
    Checks:
      • Restricted semantics in text (verbs, regions)
      • Human proximity via sensors
      • Path intersection with restricted regions (if adapter supports it)
    """
    last_warn = {"t": 0.0}  # rate-limit console prints

    def log(msg: str) -> None:
        if not verbose:
            return
        now = time.time()
        if now - last_warn["t"] > 0.2:  # avoid spam in tight loops
            print(f"[FinalStepSafety] {msg}")
            last_warn["t"] = now

    def dynamic_safety_fn(action_text: str) -> bool:
        text = action_text.strip()
        t_low = text.lower()

        # 1) Obvious text-level hazards (verbs/semantics)
        if _contains_any(t_low, ctx.forbid_verbs):
            log(f"Blocked by forbidden verb: '{text}'")
            return False

        # 2) Region-level intent in text (e.g., 'enter emergency_exit')
        if _contains_any(t_low, (r.lower() for r in ctx.restricted_regions)):
            log(f"Blocked by restricted region mentioned in action: '{text}'")
            return False

        # 3) Human proximity (sensors)
        d = sensors.nearest_human_distance()
        if d is not None:
            if d < ctx.min_human_distance_m:
                log(f"Too close to human ({d:.2f}m < {ctx.min_human_distance_m:.2f}m).")
                return False
        else:
            if strict_unknowns:
                log("Human distance unknown in strict mode.")
                return False  # fail safe if we require this signal

        # 4) Path intersection with restricted regions (requires adapter support)
        goal_xy = _parse_goal_xy(t_low)
        if goal_xy is not None:
            try:
                if sensors.path_intersects_restricted(goal_xy, ctx.restricted_regions, ctx.keepout_margin_m):
                    log(f"Path to {goal_xy} crosses restricted region (margin {ctx.keepout_margin_m}m).")
                    return False
            except NotImplementedError:
                if strict_unknowns:
                    log("path_intersects_restricted not implemented and strict mode is on.")
                    return False

        # 5) High-speed phrasing — allow but warn (or block if you prefer)
        if _contains_any(t_low, ctx.speed_risky_adverbs):
            log("Speedy phrasing detected; consider throttling velocity limits.")

        # If nothing tripped, allow
        return True

    return dynamic_safety_fn
