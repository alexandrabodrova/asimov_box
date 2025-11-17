
# adapters/examples_loader.py
# Load user scene modules that export get_mcqa_examples().

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

def load_examples_from_module(module_path: str) -> List[Dict[str, Any]]:
    mod = import_module_from_path(module_path)
    if not hasattr(mod, "get_mcqa_examples"):
        raise AttributeError(
            f"{module_path} must define get_mcqa_examples()->List[dict]."
        )
    data = mod.get_mcqa_examples()
    if not isinstance(data, list):
        raise TypeError("get_mcqa_examples() must return a list of dicts.")
    # Light validation
    for i, ex in enumerate(data):
        if "id" not in ex or "context" not in ex or "options" not in ex:
            raise ValueError(f"Example at index {i} missing required keys.")
        if not isinstance(ex["options"], list) or not ex["options"]:
            raise ValueError(f"Example {ex.get('id','<no id>')} has empty options.")
    return data
