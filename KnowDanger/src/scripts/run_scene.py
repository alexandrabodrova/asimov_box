
"""
CLI to run a given scene module with KnowDanger orchestrator and print a compact report.

Usage:
    python scripts/run_scene.py --scene example1_hazard_lab
    python scripts/run_scene.py --scene example2_breakroom
    python scripts/run_scene.py --scene example3_photonics
"""
import argparse, importlib, json
from knowdanger.core.knowdanger_core import KnowDanger
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, required=True, help="Scene module name under /scenes (e.g., example1_hazard_lab)")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--ask-thresh", type=float, default=0.6)
    args = ap.parse_args()

    kd = KnowDanger()
    kd.cfg.alpha = args.alpha
    kd.cfg.ask_threshold_confidence = args.ask_thresh

    mod = importlib.import_module(f"scenes.{args.scene}")
    scene = mod.make_scene()
    plans = mod.make_plans()

    results = []
    for p in plans:
        pa = kd.run(scene, p)
        results.append({
            "plan": p.name,
            "prompt": p.user_prompt,
            "overall": pa.overall.label,
            "why": pa.overall.why,
            "steps": [
                {
                    "action": s.step.action, "rg": s.roboguard.label, "kn": (s.knowno.label if s.knowno else None),
                    "final": s.final.label, "rg_why": s.roboguard.why, "kn_why": (s.knowno.why if s.knowno else None)
                } for s in pa.steps
            ]
        })

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
