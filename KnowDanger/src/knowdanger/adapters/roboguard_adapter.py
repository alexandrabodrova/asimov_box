from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import importlib
import inspect

Action = Union[str, Tuple[str, str]]  # "verb(object)" or ("verb","object")

def action_to_text(a: Action) -> str:
    if isinstance(a, str):
        s = a.strip()
        if "(" in s and s.endswith(")"):
            return s
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

@dataclass
class RGResult:
    ok: bool
    per_step: List[Tuple[str, bool]]
    raw: Any = None

class RoboGuardAdapter:
    """Thin adapter over the upstream RoboGuard package."""

    VERB_MAP = {
        "navigate": "go_to", "approach": "go_to",
        "enter": "enter", "block": "block", "collide": "collide",
        "handle": "pick", "pick": "pick",
        "detonate": "detonate",
        "search": "search", "inspect": "inspect", "check": "inspect",
        "scan": "scan", "count": "count",
    }

    def __init__(self, rules: Optional[Sequence[str]] = None) -> None:
        self._pkg = importlib.import_module("roboguard")
        self._rg = self._maybe_construct_guard()
        self._scene: Optional[Dict[str, Any]] = None
        self._rules: List[str] = list(rules or [])
        self._specs: Any = None
        self._specs_compiled: bool = False
        self._diag: List[str] = []

    # ---------- public ----------

    def fit(self, scene_graph: Dict[str, Any], rules: Optional[Sequence[str]] = None) -> int:
        self._scene = scene_graph or {}
        if rules is not None:
            self._rules = list(rules)

        tried: List[Tuple[str, str]] = []

        # ContextualGrounding + ControlSynthesis 
        cg = self._maybe_get("generator", "ContextualGrounding")
        cs = self._maybe_get("synthesis", "ControlSynthesis")
        if cg and cs:
            try:
                cg_inst = cg(rules="\n".join(self._rules) if self._rules else None)
                ctx_specs = cg_inst.build_contextual_specs(self._scene) if hasattr(cg_inst, "build_contextual_specs") else cg_inst(self._scene)
                syn = cs(ctx_specs)
                self._specs = getattr(syn, "specs", ctx_specs)
                self._specs_compiled = True
                self._push_state_into_rg()
                self._diag.append("[RG] ContextualGrounding + ControlSynthesis succeeded")
                return self.specs_count
            except Exception as e:
                tried.append(("ContextualGrounding/ControlSynthesis", f"{type(e).__name__}:{e}"))

        # Fallback: scan roboguard.* for synth/context functions
        for mod_name in ("synthesis", "generator", ""):
            try:
                mod = importlib.import_module(f"roboguard.{mod_name}") if mod_name else self._pkg
            except Exception:
                continue
            for attr in dir(mod):
                name = attr.lower()
                if not any(tok in name for tok in ("synth", "spec", "context")):
                    continue
                fn = getattr(mod, attr)
                if not callable(fn):
                    continue
                for args in ((self._scene, self._rules), (self._scene,), (self._rules,)):
                    try:
                        out = fn(*args)
                        if self._looks_compiled(out):
                            self._specs, self._specs_compiled = out, True
                            self._diag.append(f"[RG] roboguard.{mod_name}.{attr}{inspect.signature(fn)} worked")
                            self._push_state_into_rg()
                            return self.specs_count
                    except TypeError:
                        continue
                    except Exception as e:
                        tried.append((f"roboguard.{mod_name}.{attr}", f"{type(e).__name__}:{e}"))

        self._push_state_into_rg()
        self._diag.append("[RoboGuardAdapter] WARNING: could not synthesize contextual specs; guard may be vacuous.")
        if tried:
            self._diag.append("  Tried: " + "; ".join([f"{m}({why})" for m, why in tried]))
        return 0

    @property
    def diagnostics(self) -> List[str]:
        return list(self._diag)

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

    def check_plan_raw(self, plan: Sequence[Action]) -> Any:
        plan_strs = self._normalize_plan(plan)
        for m in ("validate_plan", "validate_action_sequence", "check_plan", "check"):
            if hasattr(self._rg, m):
                try:
                    return getattr(self._rg, m)(plan_strs)
                except Exception:
                    pass
        for mod_name in ("synthesis", "core", ""):
            try:
                mod = importlib.import_module(f"roboguard.{mod_name}") if mod_name else self._pkg
            except Exception:
                continue
            for attr in dir(mod):
                if "validate" in attr.lower() or "check" in attr.lower():
                    fn = getattr(mod, attr)
                    if callable(fn):
                        try:
                            return fn(plan_strs)
                        except Exception:
                            continue
        return None

    def check_plan(self, plan: Sequence[Action]) -> Tuple[bool, List[Tuple[str, bool]]]:
        plan_strs = self._normalize_plan(plan)
        for m in ("validate_plan", "validate_action_sequence", "check_plan"):
            if hasattr(self._rg, m):
                try:
                    raw = getattr(self._rg, m)(plan_strs)
                    ok, per_step = self._coerce_result(raw, plan_strs)
                    return ok, per_step
                except Exception:
                    continue
        raw = self.check_plan_raw(plan_strs)
        ok, per_step = self._coerce_result(raw, plan_strs)
        if per_step:
            return ok, per_step
        harmful = ("collide(", "detonate(", "handle(explosive", "block(exit", "restricted_area", "near_person")
        per_step = [(a, not any(tok in a for tok in harmful)) for a in plan_strs]
        return all(ok for _, ok in per_step), per_step

    # ---------- helpers ----------

    def _normalize_plan(self, plan: Sequence[Action]) -> List[str]:
        out: List[str] = []
        for a in plan:
            s = action_to_text(a)
            verb = s.split("(", 1)[0].strip().lower()
            norm = self.VERB_MAP.get(verb, verb)
            if norm != verb:
                s = norm + "(" + s.split("(", 1)[1]
            out.append(s)
        return out

    def _maybe_construct_guard(self) -> Any:
        candidates = [("roboguard", "RoboGuard"),
                      ("roboguard.core", "RoboGuard"),
                      ("roboguard.roboguard", "RoboGuard")]
        for mod_name, cls_name in candidates:
            try:
                mod = importlib.import_module(mod_name)
                RG = getattr(mod, cls_name, None)
                if inspect.isclass(RG):
                    return RG()
            except Exception:
                continue
        return self._pkg  # fallback namespace

    def _maybe_get(self, submodule: str, attr: str) -> Optional[Any]:
        try:
            mod = importlib.import_module(f"roboguard.{submodule}")
            return getattr(mod, attr, None)
        except Exception:
            return None

    def _looks_compiled(self, out: Any) -> bool:
        if out is None:
            return False
        if hasattr(out, "specs") or hasattr(out, "automaton") or hasattr(out, "ltl"):
            return True
        if isinstance(out, list) and all(isinstance(s, str) for s in out):
            logical = (" G ", " F ", " X ", " U ", "->", "<>", "[]", "(!", "&", "|")
            return any(any(tok in (" " + s + " ") for tok in logical) for s in out)
        return not isinstance(out, list)

    def _push_state_into_rg(self) -> None:
        target = self._rg
        if hasattr(target, "set_rules"):
            try: target.set_rules(self._rules); self._diag.append("[RG] set_rules OK")
            except Exception: pass
        elif hasattr(target, "rules"):
            try: setattr(target, "rules", self._rules); self._diag.append("[RG] rules attr set")
            except Exception: pass

        if hasattr(target, "set_scene_graph"):
            try: target.set_scene_graph(self._scene); self._diag.append("[RG] set_scene_graph OK")
            except Exception: pass
        elif hasattr(target, "scene_graph"):
            try: setattr(target, "scene_graph", self._scene); self._diag.append("[RG] scene_graph attr set")
            except Exception: pass

        if self._specs is not None:
            if hasattr(target, "set_specs"):
                try: target.set_specs(self._specs); self._diag.append("[RG] set_specs OK")
                except Exception: pass
            elif hasattr(target, "specs"):
                try: setattr(target, "specs", self._specs); self._diag.append("[RG] specs attr set")
                except Exception: pass

    def _coerce_result(self, raw: Any, plan_strs: Sequence[str]) -> Tuple[bool, List[Tuple[str, bool]]]:
        if raw is None:
            return False, []
        if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], bool):
            ok, steps = raw
            if isinstance(steps, list) and steps and isinstance(steps[0], tuple):
                return ok, [(str(a), bool(b)) for a, b in steps]
        if isinstance(raw, list) and raw and all(isinstance(x, bool) for x in raw) and len(raw) == len(plan_strs):
            return all(raw), list(zip(plan_strs, map(bool, raw)))
        if isinstance(raw, dict):
            steps = [(a, bool(raw.get(a, False))) for a in plan_strs]
            return all(ok for _, ok in steps), steps
        return False, [(a, False) for a in plan_strs]
