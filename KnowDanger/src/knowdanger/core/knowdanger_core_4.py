# knowdanger_core.py
from __future__ import annotations
import importlib
import inspect
import math
from dataclasses import dataclass
import pkgutil
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# ---------------- Types & Utilities ----------------
Action = Union[str, Tuple[str, str]]  # "verb(object)" or ("verb","object")

def action_to_text(a: Action) -> str:
    if isinstance(a, str):
        s = a.strip()
        if "(" in s and s.endswith(")"):
            return s
        if ":" in s:
            v, x = s.split(":", 1)
            return f"{v.strip()}({x.strip()})"
        if " " in s:
            v, x = s.split(" ", 1)
            return f"{v.strip()}({x.strip()})"
        return f"{s}()"
    if isinstance(a, tuple) and len(a) == 2:
        v, x = a
        return f"{str(v).strip()}({str(x).strip()})"
    raise TypeError(f"Unsupported action type: {type(a)}")

def plan_to_strings(plan: Sequence[Action]) -> List[str]:
    return [action_to_text(x) for x in plan]

# ---------------- Default rules aligned with smoke set ----------------
RULES_DEFAULT: List[str] = [
    "Never collide with a person.",
    "Never approach a person at high speed.",
    "Never handle or transport explosives.",
    "Never detonate any explosive device.",
    "Never block exits or egress routes.",
    "Never enter a restricted area.",
]

# ---------------- RoboGuard Adapter ----------------
from RoboGuard1 import RoboGuard
from asimov_box.KnowDanger.RoboGuard1.generator_1 import ContextualGrounding
from RoboGuard1.prompts.base import BASE_RULES
from RoboGuard1.synthesis import ControlSynthesis
# def _import_rg():
#     for mod in ("roboguard.roboguard", "roboguard.core", "roboguard", "RoboGuard1.roboguard"):
#         try:
#             m = importlib.import_module(mod)
#             if hasattr(m, "RoboGuard"):
#                 return getattr(m, "RoboGuard")
#         except Exception:
#             print(f"HERE IS THE ERROR: [RoboGuardAdapter] Could not import from {mod}")
#             continue
#     raise ImportError("Could not import RoboGuard class from {roboguard.roboguard, roboguard.core, roboguard, RoboGuard1.roboguard}")

class RoboGuardAdapter:
    """
    A thin, explicit adapter around the upstream library to avoid ambiguous control flow.

    Phase 1 (offline/once): fit(scene_graph, rules) -> compiles contextual specs
    Phase 2 (online/loop):  check_plan(plan) -> evaluates a candidate plan

    We keep an alias `validate_plan = check_plan` for compatibility, but never call
    a method with the same name as an upstream method to avoid 'loop' confusion.
    """
    VERB_MAP = {
        "navigate": "go_to", "approach": "go_to",
        "enter": "enter", "block": "block", "collide": "collide",
        "handle": "pick", "pick": "pick",
        "detonate": "detonate",
        "search": "search", "inspect": "inspect", "check": "inspect",
        "scan": "scan", "count": "count",
    }

    def __init__(self, rules: Optional[Sequence[str]] = None) -> None:
        RG = RoboGuard
        self._rg = RG()
        self._scene: Optional[Dict[str, Any]] = None
        self._rules: List[str] = list(rules) if rules is not None else []
        self._specs: Any = None
        self._specs_compiled: bool = False
        self._synthesis_trace: List[str] = []
        self._validate_sig: Optional[str] = None

    # ---------- diagnostics ----------
    def _record(self, msg: str) -> None:
        self._synthesis_trace.append(msg)
        print(msg)

    def _list_methods(self, obj) -> List[Tuple[str, str]]:
        out = []
        for name in dir(obj):
            if name.startswith("_"): continue
            try:
                attr = getattr(obj, name)
                if callable(attr):
                    try:
                        sig = str(inspect.signature(attr))
                    except Exception:
                        sig = "(?)"
                    out.append((name, sig))
            except Exception:
                continue
        return sorted(out)

    def _list_submodules(self, pkg) -> List[str]:
        try:
            paths = pkg.__path__
        except Exception:
            return []
        names = []
        for m in pkgutil.iter_modules(paths):
            names.append(m.name)
        return names

    # ---------- public ----------
    def fit(self, scene_graph: Dict[str, Any], rules: Optional[Sequence[str]] = None, *, cot: bool = True) -> int:
        self._scene = scene_graph or {}
        if rules is not None:
            self._rules = list(rules)

        # reset / update_context
        if hasattr(self._rg, "reset"):
            try: self._rg.reset()
            except Exception: pass
        if hasattr(self._rg, "update_context"):
            try:
                self._rg.update_context(self._scene)
                self._record("[RG] update_context(scene_graph) ok")
            except Exception as e:
                self._record(f"[RG] update_context failed: {e!r}")

        # 0) show available methods on the class
        meths = self._list_methods(self._rg)
        self._record("[RG] methods: " + ", ".join([f"{n}{s}" for (n,s) in meths]))
        if hasattr(self._rg, "validate_plan"):
            try:
                self._validate_sig = str(inspect.signature(self._rg.validate_plan))
            except Exception:
                self._validate_sig = "(signature unavailable)"
            self._record(f"[RG] validate_plan signature: {self._validate_sig}")

        # 1) try on-class synthesis/contextualization methods with many arg shapes
        self._specs = None
        self._specs_compiled = False
        synth_names = ("generate_specs", "contextualize_rules", "build_specs", "compile_specs", "contextualize")
        arg_sets = [
            ( (self._scene, self._rules, cot), {} ),
            ( (self._scene, self._rules), {"chain_of_thought": cot} ),
            ( (), {"scene_graph": self._scene, "rules": self._rules, "chain_of_thought": cot} ),
            ( (), {"semantic_graph": self._scene, "rules": self._rules, "chain_of_thought": cot} ),
            ( (self._rules,), {"chain_of_thought": cot} ),
            ( (self._rules,), {} ),
        ]
        for name in synth_names:
            if not hasattr(self._rg, name): 
                continue
            fn = getattr(self._rg, name)
            for args, kwargs in arg_sets:
                try:
                    out = fn(*args, **kwargs)
                    if self._looks_compiled(out):
                        self._specs, self._specs_compiled = out, True
                        self._record(f"[RG] {name}{inspect.signature(fn)} succeeded with args={list(kwargs) or 'positional'}")
                        self._push_state_into_rg()
                        return self.specs_count
                    else:
                        self._record(f"[RG] {name} returned non-compiled object: {type(out).__name__}")
                except TypeError as e:
                    self._record(f"[RG] {name} TypeError: {e}")
                except Exception as e:
                    self._record(f"[RG] {name} Exception: {e!r}")

        # 2) try module-level functions inside roboguard.*
        try:
            pkg = importlib.import_module("roboguard")
            submods = self._list_submodules(pkg)
            self._record("[RG] submodules: " + ", ".join(submods))
            for mname in submods:
                if not any(tok in mname.lower() for tok in ("spec", "synth", "context")):
                    continue
                mod = importlib.import_module(f"roboguard.{mname}")
                for attr in dir(mod):
                    if any(tok in attr.lower() for tok in ("spec", "synth", "context")):
                        fn = getattr(mod, attr)
                        if not callable(fn): continue
                        try:
                            out = fn(self._scene, self._rules)  # most likely
                            if self._looks_compiled(out):
                                self._specs, self._specs_compiled = out, True
                                self._record(f"[RG] roboguard.{mname}.{attr} succeeded")
                                self._push_state_into_rg()
                                return self.specs_count
                        except TypeError as e:
                            self._record(f"[RG] roboguard.{mname}.{attr} TypeError: {e}")
                        except Exception as e:
                            self._record(f"[RG] roboguard.{mname}.{attr} Exception: {e!r}")
        except Exception as e:
            self._record(f"[RG] submodule probing failed: {e!r}")

        # 3) no compiled specs; push at least rules & scene so a stateful checker can use them
        self._push_state_into_rg()
        self._record("[RoboGuardAdapter] WARNING: could not synthesize contextual specs; guard may be vacuous.")
        return 0

    def specs_status(self) -> str:
        if self._specs_compiled and self._specs is not None:
            return f"compiled:{self.specs_count}"
        return "none"

    @property
    def specs_count(self) -> int:
        if not self._specs_compiled or self._specs is None:
            return 0
        try:
            return len(self._specs)
        except Exception:
            try:
                return len(getattr(self._specs, "specs", []))
            except Exception:
                return 1

    def check_plan_raw(self, plan) -> Any:
        plan_strs = self._normalize_plan(plan_to_strings(plan))
        return self._call_rg_validate(plan_strs)

    def check_plan(self, plan) -> Tuple[bool, List[Tuple[str,bool]]]:
        plan_strs = self._normalize_plan(plan_to_strings(plan))
        raw = self._call_rg_validate(plan_strs)

        # Normalize
        if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], bool):
            feasible, details = raw
            if (isinstance(details, list) and details and isinstance(details[0], tuple)
                and len(details[0]) >= 2 and isinstance(details[0][1], (bool, int))):
                step_ok = [(str(a), bool(ok)) for (a, ok, *_) in details]
                return all(ok for _, ok in step_ok), step_ok
            # Feasible w/o trace => fail-closed on original sequence
            self._record("[RoboGuardAdapter] Feasible synthesis but no per-step trace; treating original steps as UNSAFE.")
            return False, [(a, False) for a in plan_strs]
        if isinstance(raw, list) and raw and isinstance(raw[0], tuple) and len(raw[0]) >= 2:
            step_ok = [(str(a), bool(ok)) for (a, ok, *_) in raw]
            return all(ok for _, ok in step_ok), step_ok
        if isinstance(raw, list) and all(isinstance(x, (bool,int)) for x in raw):
            step_ok = list(zip(plan_strs, [bool(x) for x in raw]))
            return all(ok for _, ok in step_ok), step_ok
        if isinstance(raw, bool):
            self._record("[RoboGuardAdapter] Upstream returned bare bool; failing closed.")
            return False, [(p, False) for p in plan_strs]
        self._record("[RoboGuardAdapter] Unexpected validate_plan return; failing closed.")
        return False, [(p, False) for p in plan_strs]

    validate_plan = check_plan  # compatibility

    # ---------- helpers ----------
    def _looks_compiled(self, out: Any) -> bool:
        if out is None:
            return False
        if hasattr(out, "specs") or hasattr(out, "automaton") or hasattr(out, "ltl"):
            return True
        if isinstance(out, list) and all(isinstance(s, str) for s in out):
            logical_tokens = ("G ", "F ", "X ", " U ", "->", "<>", "[]", "(!", "&", "|")
            return any(any(tok in s for tok in logical_tokens) for s in out)
        return not isinstance(out, list)

    def _push_state_into_rg(self) -> None:
        # Many RG snapshots are stateful: set rules, scene, specs directly into the object
        if hasattr(self._rg, "set_rules"):
            try: self._rg.set_rules(self._rules); self._record("[RG] set_rules(rules) ok")
            except Exception as e: self._record(f"[RG] set_rules failed: {e!r}")
        elif hasattr(self._rg, "rules"):
            try: setattr(self._rg, "rules", self._rules); self._record("[RG] rules attribute set")
            except Exception as e: self._record(f"[RG] rules attr set failed: {e!r}")

        if hasattr(self._rg, "set_scene_graph"):
            try: self._rg.set_scene_graph(self._scene); self._record("[RG] set_scene_graph(scene) ok")
            except Exception as e: self._record(f"[RG] set_scene_graph failed: {e!r}")
        elif hasattr(self._rg, "scene_graph"):
            try: setattr(self._rg, "scene_graph", self._scene); self._record("[RG] scene_graph attribute set")
            except Exception as e: self._record(f"[RG] scene_graph attr set failed: {e!r}")

        if self._specs is not None:
            if hasattr(self._rg, "set_specs"):
                try: self._rg.set_specs(self._specs); self._record("[RG] set_specs(specs) ok")
                except Exception as e: self._record(f"[RG] set_specs failed: {e!r}")
            elif hasattr(self._rg, "specs"):
                try: setattr(self._rg, "specs", self._specs); self._record("[RG] specs attribute set")
                except Exception as e: self._record(f"[RG] specs attr set failed: {e!r}")

    def _normalize_plan(self, plan_strs: List[str]) -> List[str]:
        out = []
        for s in plan_strs:
            if "(" in s and s.endswith(")"):
                v, arg = s[:-1].split("(", 1)
                v = self.VERB_MAP.get(v.strip().lower(), v.strip())
                arg = self._ground_object(arg.strip())
                out.append(f"{v}({arg})")
            else:
                v = self.VERB_MAP.get(s.strip().lower(), s.strip().lower())
                out.append(f"{v}()")
        return out

    def _ground_object(self, obj: str) -> str:
        o = obj.lower().replace(" ", "_")
        aliases = {
            "near_person": "person_0", "towards_person": "person_0", "person": "person_0",
            "emergency_exit": "exit_0", "exit": "exit_0",
            "restricted_area": "restricted_area_0",
            "weapon": "weapon_0", "bomb": "explosive_0", "explosive": "explosive_0",
            "chair": "chair_0", "door": "door_0", "plant": "plant_0", "scene": "scene_0",
        }
        return aliases.get(o, o)

    def _call_rg_validate(self, plan_strs: List[str]) -> Any:
        # Prefer passing specs if signature allows; otherwise rely on internal state (set via _push_state_into_rg)
        try:
            return self._rg.validate_plan(plan_strs, self._specs)
        except TypeError:
            try:
                return self._rg.validate_plan(plan=plan_strs, specs=self._specs)
            except Exception:
                pass
        # If we get here, checker is stateful; we already pushed rules/scene/specs (if any)
        self._record("[RoboGuardAdapter] calling validate_plan(plan) (stateful mode).")
        return self._rg.validate_plan(plan_strs)


# def _import_rg():
#     # Try likely module paths for the class "RoboGuard"
#     for mod in ("roboguard.roboguard", "roboguard.core", "roboguard", "RoboGuard1.roboguard"):
#         try:
#             m = importlib.import_module(mod)
#             if hasattr(m, "RoboGuard"):
#                 return getattr(m, "RoboGuard")
#         except Exception:
#             continue
#     raise ImportError("Could not import RoboGuard (tried roboguard.roboguard, roboguard.core, roboguard, RoboGuard1.roboguard).")

# class RoboGuardAdapter:
#     """
#     A thin, explicit adapter around the upstream library to avoid ambiguous control flow.

#     Phase 1 (offline/once): fit(scene_graph, rules) -> compiles contextual specs
#     Phase 2 (online/loop):  check_plan(plan) -> evaluates a candidate plan

#     We keep an alias `validate_plan = check_plan` for compatibility, but never call
#     a method with the same name as an upstream method to avoid 'loop' confusion.
#     """
#     # Canonical verb map for grounding free-form tokens
#     VERB_MAP = {
#         "navigate": "go_to",
#         "approach": "go_to",
#         "enter": "enter",
#         "block": "block",
#         "collide": "collide",
#         "handle": "pick",
#         "pick": "pick",
#         "detonate": "detonate",
#         "search": "search",
#         "inspect": "inspect",
#         "check": "inspect",
#         "scan": "scan",
#         "count": "count",
#     }

#     def __init__(self, rules: Optional[Sequence[str]] = None) -> None:
#         RG = _import_rg()
#         self._rg = RG()
#         self._scene: Optional[Dict[str, Any]] = None
#         self._rules: List[str] = list(rules) if rules is not None else list(RULES_DEFAULT)
#         self._specs: Any = None
#         self._specs_compiled: bool = False  # True only if we have contextualized/compiled specs

#     # ---------- Public API ----------
#     def fit(self, scene_graph: Dict[str, Any], rules: Optional[Sequence[str]] = None, *, cot: bool = True) -> int:
#         """Compile contextual specs from (scene_graph, rules). Returns the number of resulting specs
#         or 0 if compilation failed. Sets self._specs and self._specs_compiled."""
#         self._scene = scene_graph or {}
#         if rules is not None:
#             self._rules = list(rules)

#         # Give context to RG if exposed
#         if hasattr(self._rg, "reset"):
#             try: self._rg.reset()
#             except Exception: pass
#         if hasattr(self._rg, "update_context"):
#             try: self._rg.update_context(self._scene)
#             except Exception as e:
#                 print(f"[RoboGuardAdapter] update_context failed: {e}")

#         # Try to synthesize/contextualize
#         self._specs = None
#         self._specs_compiled = False

#         spec_methods = ("generate_specs", "contextualize_rules", "build_specs", "compile_specs")
#         tried = []

#         # Try multiple arg/kw patterns
#         for m in spec_methods:
#             if not hasattr(self._rg, m):
#                 continue
#             fn = getattr(self._rg, m)
#             arg_options = [
#                 ( (self._scene, self._rules, cot), {} ),
#                 ( (self._scene, self._rules), {"chain_of_thought": cot} ),
#                 ( (self._rules, self._scene), {"chain_of_thought": cot} ),
#                 ( (), {"scene_graph": self._scene, "rules": self._rules, "chain_of_thought": cot} ),
#                 ( (), {"semantic_graph": self._scene, "rules": self._rules, "chain_of_thought": cot} ),
#                 ( (), {"graph": self._scene, "rules": self._rules, "chain_of_thought": cot} ),
#                 ( (self._rules,), {"chain_of_thought": cot} ),
#                 ( (self._rules,), {} ),
#             ]
#             for args, kwargs in arg_options:
#                 try:
#                     out = fn(*args, **kwargs)
#                     if self._is_compiled_specs(out):
#                         self._specs = out
#                         self._specs_compiled = True
#                         return self.specs_count
#                     else:
#                         tried.append((m, "returned-uncompiled-or-empty"))
#                 except TypeError as e:
#                     tried.append((m, f"TypeError:{e}"))
#                 except Exception as e:
#                     tried.append((m, f"Exception:{e}"))
#         # No luck
#         print("[RoboGuardAdapter] WARNING: could not synthesize contextual specs; guard may be vacuous.")
#         if tried:
#             print("  Tried methods:", "; ".join([f"{m}({why})" for m, why in tried]))
#         return 0

#     def specs_status(self) -> str:
#         if self._specs_compiled and self._specs is not None:
#             return f"compiled:{self.specs_count}"
#         if self._specs is None:
#             return "none"
#         # Some object exists but not confidently compiled (e.g., list of raw rules)
#         return "uncompiled"

#     @property
#     def specs_count(self) -> int:
#         if not self._specs_compiled or self._specs is None:
#             return 0
#         try:
#             return len(self._specs)
#         except Exception:
#             try:
#                 return len(getattr(self._specs, "specs", []))
#             except Exception:
#                 return 1

#     def check_plan_raw(self, plan: Sequence[Action]) -> Any:
#         """Forward the plan to upstream (with specs if available) and return raw upstream output."""
#         plan_strs = self._normalize_plan(plan_to_strings(plan))
#         return self._call_rg_validate(plan_strs)

#     def check_plan(self, plan: Sequence[Action]) -> Tuple[bool, List[Tuple[str, bool]]]:
#         """Normalize upstream returns to (is_safe, [(action, ok), ...]). Fail-closed if ambiguous."""
#         plan_strs = self._normalize_plan(plan_to_strings(plan))
#         raw = self._call_rg_validate(plan_strs)

#         # Normalize
#         step_ok: List[Tuple[str, bool]] = []
#         # A) (bool, details)
#         if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], bool):
#             feasible, details = raw
#             if (isinstance(details, list) and details and isinstance(details[0], tuple)
#                 and len(details[0]) >= 2 and isinstance(details[0][1], (bool, int))):
#                 step_ok = [(str(a), bool(ok)) for (a, ok, *_) in details]
#                 return all(ok for _, ok in step_ok), step_ok
#             # Try to obtain a repaired/safe plan if available; otherwise fail-closed
#             for m in ("enforce", "repair_plan", "synthesize_safe_plan", "resolve_conflicts"):
#                 if hasattr(self._rg, m):
#                     try:
#                         safe_plan = getattr(self._rg, m)(plan_strs, self._specs)
#                         if isinstance(safe_plan, list):
#                             safe_set = set(safe_plan)
#                             step_ok = [(a, a in safe_set) for a in plan_strs]
#                             return all(ok for _, ok in step_ok), step_ok
#                     except Exception:
#                         pass
#             print("[RoboGuardAdapter] Feasible synthesis but no per-step trace; treating original steps as UNSAFE.")
#             return False, [(a, False) for a in plan_strs]

#         # B) list[(action,bool)]
#         if isinstance(raw, list) and raw and isinstance(raw[0], tuple) and len(raw[0]) >= 2 and isinstance(raw[0][1], (bool, int)):
#             step_ok = [(str(a), bool(ok)) for (a, ok, *_) in raw]
#             return all(ok for _, ok in step_ok), step_ok

#         # C) list[bool]
#         if isinstance(raw, list) and all(isinstance(x, (bool, int)) for x in raw):
#             step_ok = list(zip(plan_strs, [bool(x) for x in raw]))
#             return all(ok for _, ok in step_ok), step_ok

#         # D) single bool -> ambiguous; fail-closed
#         if isinstance(raw, bool):
#             print("[RoboGuardAdapter] Upstream returned bare bool; failing closed (no per-step trace).")
#             return (False if raw else False), [(p, False) for p in plan_strs]

#         print("[RoboGuardAdapter] Unexpected validate_plan return; marking all steps as UNSAFE.")
#         return False, [(p, False) for p in plan_strs]

#     # Compatibility alias if external code expects 'validate_plan'
#     validate_plan = check_plan

#     # ---------- Internals ----------
#     def _is_compiled_specs(self, out: Any) -> bool:
#         """Heuristic: treat as compiled only if it is not trivially equal to the raw rules,
#         and looks like a set of logical specs (or an object with a 'specs' attribute)."""
#         if out is None:
#             return False
#         # Structured object with specs field
#         if hasattr(out, "specs") or hasattr(out, "ltl") or hasattr(out, "automaton"):
#             return True
#         # List[str] may be LTL; accept only if it doesn't equal the raw rules and looks logical
#         if isinstance(out, list) and all(isinstance(s, str) for s in out):
#             if out == self._rules:
#                 return False
#             logical_tokens = ("G ", "F ", "X ", " U ", "->", "<>", "[]", "(!", "&", "|")
#             if any(any(tok in s for tok in logical_tokens) for s in out):
#                 return True
#             return False
#         # Non-list objects -> assume compiled
#         return not isinstance(out, list)

#     def _normalize_plan(self, plan_strs: List[str]) -> List[str]:
#         out = []
#         for s in plan_strs:
#             if "(" in s and s.endswith(")"):
#                 v, arg = s[:-1].split("(", 1)
#                 v = self.VERB_MAP.get(v.strip().lower(), v.strip())
#                 arg = self._ground_object(arg.strip())
#                 out.append(f"{v}({arg})")
#             else:
#                 v = self.VERB_MAP.get(s.strip().lower(), s.strip().lower())
#                 out.append(f"{v}()")
#         return out

#     def _ground_object(self, obj: str) -> str:
#         o = obj.lower().replace(" ", "_")
#         # Default aliases; override if your scene graph uses different IDs
#         aliases = {
#             "near_person": "person_0",
#             "towards_person": "person_0",
#             "person": "person_0",
#             "emergency_exit": "exit_0",
#             "exit": "exit_0",
#             "restricted_area": "restricted_area_0",
#             "weapon": "weapon_0",
#             "bomb": "explosive_0",
#             "explosive": "explosive_0",
#             "chair": "chair_0",
#             "door": "door_0",
#             "plant": "plant_0",
#             "scene": "scene_0",
#         }
#         return aliases.get(o, o)

#     def _call_rg_validate(self, plan_strs: List[str]) -> Any:
#         """Prefer (plan, specs); then try (plan, rules, scene), then (plan, rules), last (plan)."""
#         # (plan, specs)
#         try:
#             if self._specs_compiled and self._specs is not None:
#                 return self._rg.validate_plan(plan_strs, self._specs)
#         except TypeError:
#             try:
#                 return self._rg.validate_plan(plan=plan_strs, specs=self._specs)
#             except Exception:
#                 pass
#         # (plan, rules, scene)
#         try:
#             return self._rg.validate_plan(plan_strs, self._rules, self._scene)
#         except TypeError:
#             try:
#                 return self._rg.validate_plan(plan=plan_strs, rules=self._rules, scene_graph=self._scene)
#             except Exception:
#                 pass
#         # (plan, rules)
#         try:
#             return self._rg.validate_plan(plan_strs, self._rules)
#         except Exception:
#             pass
#         # (plan) only — warn (vacuous risk)
#         print("[RoboGuardAdapter] WARNING: calling validate_plan(plan) without specs/rules; result may be vacuous.")
#         return self._rg.validate_plan(plan_strs)

# ---------------- Minimal KnowNo-like certifier ----------------
HARMFUL_TOKENS = [
    "collide(", "detonate(", "handle(explosive", "weapon", "block(exit", "restricted_area", "near_person", "towards_person"
]

@dataclass
class Certifier:
    mode: str = "pA"
    p_threshold: float = 0.80
    tau: float = 0.7
    def score(self, a_text: str) -> float:
        a = a_text.lower()
        is_harm = any(tok in a for tok in HARMFUL_TOKENS)
        lA = -2.2 if is_harm else 2.2
        if self.mode == "gap":
            # logit margin scaled by tau
            return (lA - (-lA)) / max(self.tau, 1e-3)
        return 1.0 / (1.0 + math.exp(-(lA) / max(self.tau, 1e-3)))
    def certify(self, a_text: str) -> bool:
        pA = self.score(a_text)
        return (pA >= self.p_threshold) if self.mode != "gap" else (pA >= 0.0)

class FinalStepInterceptor:
    def intercept(self, action_text: str) -> bool:
        return True

class PlanValidator:
    def __init__(self, harmful_keywords: Optional[Sequence[str]] = None) -> None:
        self.harmful = list(harmful_keywords) if harmful_keywords is not None else HARMFUL_TOKENS
    def validate_plan1(self, plan: Sequence[Action]) -> List[Action]:
        return list(plan)
    def validate_plan(self, plan: Sequence[Action]) -> List[Action]:
        return self.validate_plan1(plan)

# ---------------- KnowDanger Orchestrator ----------------
class KnowDanger:
    def __init__(
        self,
        use_roboguard: bool = True,
        certifier_mode: str = "pA",
        p_threshold: float = 0.80,
        tau: float = 0.7,
        rules: Optional[Sequence[str]] = None,
    ) -> None:
        self.use_roboguard = use_roboguard
        self.plan_validator = PlanValidator()
        self.certifier = Certifier(mode=certifier_mode, p_threshold=p_threshold, tau=tau)
        self.final_step_interceptor = FinalStepInterceptor()
        self._rg: Optional[RoboGuardAdapter] = RoboGuardAdapter(rules=rules or RULES_DEFAULT) if use_roboguard else None

    def fit_guard(self, scene_graph: Dict[str, Any], rules: Optional[Sequence[str]] = None) -> None:
        if self._rg is None:
            return
        n = self._rg.fit(scene_graph, rules)
        print(f"[KnowDanger] RoboGuard specs_status: {self._rg.specs_status()} (count={n})")

    def execute_plan(self, plan: Sequence[Action], verbose: bool = True) -> Tuple[bool, List[Tuple[str,bool]]]:
        plan_strs = plan_to_strings(plan)

        # Gate 1: RoboGuard
        if self._rg:
            raw = self._rg.check_plan_raw(plan)
            if verbose:
                print("RG plan:", plan_strs, "→", raw)
            ok_rg, step_rg = self._rg.check_plan(plan)
            if verbose and not ok_rg:
                print("[KnowDanger] RoboGuard rejected plan:")
                for a, ok in step_rg:
                    print(f"   {a} {'✅' if ok else '❌'}")
            if not ok_rg:
                return False, step_rg
            allowed_strs = [a for a, ok in step_rg if ok]
        else:
            allowed_strs = plan_strs

        # Gate 2: trivial pre-filter
        candidate = self.plan_validator.validate_plan(allowed_strs)

        # Gate 3: Certifier + final interceptor
        final_steps: List[Tuple[str, bool]] = []
        for a in candidate:
            a_txt = action_to_text(a)
            if not self.certifier.certify(a_txt):
                if verbose: print(f"[KnowDanger] Refusing due to low confidence: {a_txt}")
                final_steps.append((a_txt, False)); continue
            ok = self.final_step_interceptor.intercept(a_txt)
            final_steps.append((a_txt, bool(ok)))
        return all(ok for _, ok in final_steps), final_steps
