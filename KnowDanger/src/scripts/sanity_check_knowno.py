
"""
Sanity checks for lang-help/knowno import surfaces.
Run with the same Python as your pipeline.
"""
import importlib, sys, json

mods = [
    "lang_help.knowno.api",
    "known.knowno.api",
    "agent.predict.conformal_predictor",
    "agent.predict.set_predictor",
]

res = {}
for m in mods:
    try:
        mod = importlib.import_module(m)
        res[m] = {"ok": True, "file": getattr(mod, "__file__", "<pkg>")}
    except Exception as e:
        res[m] = {"ok": False, "error": repr(e)}

print(json.dumps({"python": sys.executable, "results": res}, indent=2))
