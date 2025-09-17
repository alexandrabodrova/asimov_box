
"""
Benchmark: Compare KnowDanger vs. RoboGuard vs. KnowNo on all three examples.
This script treats 'roboguard' and 'lang_help' imports as optional; if present,
their native behavior will be used. Otherwise, fallback behavior keeps the
evaluation runnable (but less faithful).

Run:
    python -m tests.benchmark_knowno_roboguard --scenes example1_hazard_lab example2_breakroom example3_photonics
"""
import argparse, importlib, json
from copy import deepcopy
from knowdanger.core.knowdanger_core import KnowDanger, Config, Verdict

def evaluate_variant(scene_mod_name: str, variant: str, alpha=0.1, ask_thresh=0.7):
    mod = importlib.import_module(f"scenes.{scene_mod_name}")
    scene = mod.make_scene()
    plans = mod.make_plans()

    kd = KnowDanger(Config(alpha=alpha))
    kd.cfg.ask_threshold_confidence = ask_thresh

    # Variants:
    #   - 'roboguard': disable KnowNo (don't look at candidates)
    #   - 'knowno': disable RoboGuard (treat all rg as SAFE)
    #   - 'knowdanger': both on (default)
    def run_plan_variant(plan):
        original_steps = plan.steps
        if variant == "roboguard":
            # wipe candidates so KnowNo yields None
            plan = deepcopy(plan)
            for st in plan.steps:
                st.candidates = None
        elif variant == "knowno":
            # wrap RG bridge to always return SAFE
            kd.rg.evaluate_plan = lambda plan_, compiled: [Verdict("SAFE", "RG disabled", {}) for _ in plan_.steps]
        return kd.run(scene, plan)

    results = [run_plan_variant(p) for p in plans]
    # Metrics: unsafe block rate, uncertain ask rate, safe pass rate
    m = {"SAFE":0, "UNSAFE":0, "UNCERTAIN":0}
    for r in results:
        m[r.overall.label] += 1
    return {
        "scene": scene_mod_name,
        "variant": variant,
        "counts": m,
        "details": [
            {
                "plan": r.plan.name, "label": r.overall.label, "why": r.overall.why,
                "step_labels": [s.final.label for s in r.steps]
            } for r in results
        ]
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["example1_hazard_lab","example2_breakroom","example3_photonics"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--ask-thresh", type=float, default=0.7)
    args = ap.parse_args()

    out = []
    for sc in args.scenes:
        for v in ("roboguard","knowno","knowdanger"):
            out.append(evaluate_variant(sc, v, alpha=args.alpha, ask_thresh=args.ask_thresh))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
