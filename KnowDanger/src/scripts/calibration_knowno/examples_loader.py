
"""
Adapter to load MCQA examples from a user-provided Python module.
We expect the module to define a function:

    def get_mcqa_examples() -> list[dict]:
        return [
            {
              "id": "scene_001",
              "context": "You are in a lab with ...",
              "options": ["action A", "action B", "action C"],
              # Optional:
              # "meta": {"family": "hazard_lab"}
            },
            ...
        ]

If your existing files (example1_hazard_lab.py, etc.) don't export this,
you can either (a) add the function to them, or (b) create a thin wrapper
module that imports and exposes get_mcqa_examples().
"""

import importlib.util
from types import ModuleType
from typing import List, Dict, Any


def import_module_from_path(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("user_scenes", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_examples(module_path: str) -> List[Dict[str, Any]]:
    mod = import_module_from_path(module_path)
    if not hasattr(mod, "get_mcqa_examples"):
        raise AttributeError(
            f"{module_path} must define get_mcqa_examples()->List[dict]."
        )
    data = mod.get_mcqa_examples()
    if not isinstance(data, list):
        raise TypeError("get_mcqa_examples() must return a list of dicts.")
    return data
