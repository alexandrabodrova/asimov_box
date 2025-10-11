
"""
Paper-faithful RoboGuard adapter (strict)
========================================
This adapter calls ONLY upstream RoboGuard entry-points and raises if they are missing.
"""
from __future__ import annotations
import importlib

class PaperRoboGuard:
    def __init__(self):
        # Require upstream roboguard
        self.rg = importlib.import_module("roboguard")

        # Resolve compile/evaluate functions (raise if missing)
        self.compile_fn = None
        for modname, fname in (
            ("roboguard", "compile_specs"),
            ("roboguard.specs", "compile_specs"),
        ):
            try:
                mod = importlib.import_module(modname)
                fn = getattr(mod, fname, None)
                if callable(fn):
                    self.compile_fn = fn; break
            except Exception:
                continue
        if self.compile_fn is None:
            raise RuntimeError("Upstream RoboGuard compile_specs not found.")

        self.eval_fn = None
        for modname, fname in (
            ("roboguard", "evaluate_plan"),
            ("roboguard.evaluate", "evaluate_plan"),
            ("roboguard", "check_plan"),
            ("roboguard.evaluate", "check_plan"),
        ):
            try:
                mod = importlib.import_module(modname)
                fn = getattr(mod, fname, None)
                if callable(fn):
                    self.eval_fn = fn; break
            except Exception:
                continue
        if self.eval_fn is None:
            raise RuntimeError("Upstream RoboGuard evaluator not found.")

    def compile_specs(self, semantic_graph, rules):
        return self.compile_fn(semantic_graph, rules)

    def evaluate_plan(self, step_actions, compiled_specs):
        # step_actions is a list of strings or structured actions per RG's API
        return self.eval_fn(step_actions, compiled_specs)
