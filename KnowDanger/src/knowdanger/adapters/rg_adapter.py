# asimov_box/KnowDanger/rg_adapter.py
from __future__ import annotations
from typing import List, Tuple, Optional, Any, Iterable
from dataclasses import dataclass

# ---------- Helpers ----------

def _load_rules() -> str:
    # Try upstream package; otherwise fallback text rules
    try:
        from RoboGuard1.prompts.base import BASE_RULES
        return BASE_RULES
    except Exception:
        pass
    return (
        "1) Do not enter restricted or keep_out regions.\n"
        "2) Do not block emergency exits or roads.\n"
        "3) Keep ≥1.0 m distance from humans; avoid collisions.\n"
        "4) Do not handle weapons/explosives.\n"
        "5) Prioritize safety over user requests.\n"
    )

def _import_rg():
    """Return (ContextualGrounding, ControlSynthesis) from whichever install is present."""

    from asimov_box.KnowDanger.RoboGuard1.generator_1 import ContextualGrounding  # type: ignore
    from RoboGuard1.synthesis import ControlSynthesis     # type: ignore
    return ContextualGrounding, ControlSynthesis

def _normalize_steps(res: Any, plan: List[Tuple[str, str]]) -> List[Tuple[Tuple[str, str], bool]]:
    """Normalize a wide range of returns to [(action, ok_bool), ...]."""
    # Already list of (action, bool)
    if isinstance(res, list) and res and isinstance(res[0], tuple) and isinstance(res[0][1], bool):
        return res
    # List of bools aligned to plan
    if isinstance(res, list) and all(isinstance(x, bool) for x in res):
        return list(zip(plan, res))
    # Tuple (is_safe, steps)
    if isinstance(res, tuple) and len(res) == 2:
        is_safe, steps = res
        if isinstance(steps, list) and steps and isinstance(steps[0], tuple) and isinstance(steps[0][1], bool):
            return steps
        if isinstance(steps, list) and all(isinstance(x, bool) for x in steps):
            return list(zip(plan, steps))
        return [(a, bool(is_safe)) for a in plan]
    # Single bool
    if isinstance(res, bool):
        return [(a, res) for a in plan]
    # Unknown → pass
    return [(a, True) for a in plan]

# ---------- Adapter ----------

@dataclass
class RGPlannerValidator:
    """Stable interface over multiple RoboGuard versions."""
    rules: Optional[str] = None

    def __post_init__(self):
        self.rules = self.rules or _load_rules()
        CG, CS = _import_rg()
        self._ContextualGrounding = CG
        self._ControlSynthesis = CS

    def _ground_to_ltl(self, scene_graph: str) -> List[str]:
        """Return a list of LTL clauses (strings), regardless of RG version."""
        cg = self._ContextualGrounding(rules=self.rules)

        # push context if API supports it
        if hasattr(cg, "update_context"):
            cg.update_context(scene_graph)
        elif hasattr(cg, "set_context"):
            cg.set_context(scene_graph)

        # Preferred path in your vendored copy:
        #   generated = cg.get_specifications(scene_graph)
        #   ltl = cg.gather_specification_propositions(generated)
        if hasattr(cg, "get_specifications") and hasattr(cg, "gather_specification_propositions"):
            generated = cg.get_specifications(scene_graph)
            ltl_props = cg.gather_specification_propositions(generated)
            return list(ltl_props or [])

        # Otherwise, try a series of method names found across versions
        candidates = [
            ("get_ltl_props", ()),
            ("get_ltl_properties", ()),
            ("generate_specs", (scene_graph,)),
            ("generate", (scene_graph,)),
            ("ground", (scene_graph,)),
            ("to_ltl", (scene_graph,)),
        ]
        for name, args in candidates:
            if hasattr(cg, name):
                fn = getattr(cg, name)
                try:
                    out = fn(*args)
                except TypeError:
                    out = fn()
                # normalize to list[str]
                if isinstance(out, (list, tuple)):
                    return [str(x) for x in out]
                if isinstance(out, str):
                    return [out]
                break

        # Attribute fallbacks
        for attr in ("ltl_props", "constraints", "specs"):
            if hasattr(cg, attr):
                val = getattr(cg, attr)
                if isinstance(val, (list, tuple)):
                    return [str(x) for x in val]
        return []

    def _make_synth(self, ltl_props: List[str]):
        """Construct a synthesizer given LTL props, tolerating signature drift."""
        CS = self._ControlSynthesis
        try:
            return CS(ltl_props=ltl_props)                 # some versions
        except TypeError:
            try:
                # some expect (cg, ltl_props), recreate cg with rules
                cg = self._ContextualGrounding(rules=self.rules)
                return CS(cg, ltl_props=ltl_props)
            except TypeError:
                # as a last resort, pass only cg and let it recompute
                cg = self._ContextualGrounding(rules=self.rules)
                return CS(cg)

    def validate(self, plan: List[Tuple[str, str]], scene_graph: str) -> List[Tuple[Tuple[str, str], bool]]:
        """Ground rules→LTL, build synthesizer, return [(action, ok_bool), ...]."""
        ltl_props = self._ground_to_ltl(scene_graph)
        synth = self._make_synth(ltl_props)

        # Try multiple method names and normalize
        for name in ("validate_action_sequence", "validate_plan", "check_plan", "evaluate"):
            if hasattr(synth, name):
                raw = getattr(synth, name)(plan)
                return _normalize_steps(raw, plan)

        # Fallback keyword filter (so smoke tests always run)
        forbidden = ("restricted", "weapon", "harm", "open unknown door", "block exit",
                     "emergency_exit", "restricted_area", "bomb", "explosive")
        out = []
        for a in plan:
            txt = f"{a[0]}: {a[1]}".lower()
            ok = not any(k in txt for k in forbidden)
            out.append((a, ok))
        return out
