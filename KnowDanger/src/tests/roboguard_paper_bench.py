
"""
Run RoboGuard on its original paper test suites, and (optionally) overlay KnowDanger and KnowNo.

This runner is adapter-based because different versions of the RoboGuard repo expose their
paper tests differently. You must supply either:
  (A) a Python loader function via --suite-loader 'module.submod:function', which returns a
      list of suite dicts in a normalized format (see below), or
  (B) a suite root path via --suite-root with JSON/YAML files you provide in the normalized format.

Normalized suite format per scene:
{
  "name": "SceneName",
  "semantic_graph": {...},           # dict
  "rules": ["G(...)", "..."],        # list[str]
  "plans": [
    {
      "name": "PlanName",
      "user_prompt": "Prompt",
      "steps": [
        {"action": "place", "params": {"x": "a", "s": "b"}, "candidates": [["opt1", 0.6], ["opt2", 0.4]], "meta": {...}},
        ...
      ]
    },
    ...
  ]
}

Usage examples:
  # Loader function path (recommended once you write a tiny adapter in your RG repo)
  python -m tests.roboguard_paper_bench --suite-loader roboguard.paper_suites:get_suites

  # Or: supply a directory with normalized JSON files (one file per scene)
  python -m tests.roboguard_paper_bench --suite-root /path/to/rg_suites_json

Output: report + CSVs similar to benchmark scripts.
"""
import argparse, importlib, json, os, sys, csv, datetime, glob
from typing import List, Dict, Any, Optional
from knowdanger.core.knowdanger_core import KnowDanger, Config, Scene, PlanCandidate, Step, Verdict

def load_suites_from_loader(loader_path: str) -> List[Dict[str, Any]]:
    mod_path, fn_name = loader_path.split(":")
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, fn_name)
    suites = fn()
    if not isinstance(suites, (list,tuple)):
        raise RuntimeError("Suite loader must return a list of scenes.")
    return suites

def load_suites_from_dir(root: str) -> List[Dict[str, Any]]:
    out = []
    for p in glob.glob(os.path.join(root, "*.json")) + glob.glob(os.path.join(root, "*.yaml")) + glob.glob(os.path.join(root, "*.yml")):
        with open(p, "r") as f:
            if p.endswith(".json"):
                scene = json.load(f)
            else:
                try:
                    import yaml
                except Exception as e:
                    raise RuntimeError(f"YAML file found but PyYAML not installed: {p}")
                scene = yaml.safe_load(f)
        out.append(scene)
    if not out:
        raise RuntimeError(f"No suite files found in {root}")
    return out

def to_scene(scene_dict: Dict[str, Any]) -> Scene:
    return Scene(
        name=scene_dict["name"],
        semantic_graph=scene_dict["semantic_graph"],
        rules=scene_dict["rules"],
        env_params=scene_dict.get("env_params", {}),
        helpers={}
    )

def to_plan(plan_dict: Dict[str, Any]) -> PlanCandidate:
    steps = []
    for sd in plan_dict["steps"]:
        steps.append(Step(
            action=sd["action"],
            params=sd.get("params", {}),
            candidates=sd.get("candidates", None),
            meta=sd.get("meta", {}),
        ))
    return PlanCandidate(
        name=plan_dict["name"],
        user_prompt=plan_dict.get("user_prompt", ""),
        steps=steps
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-loader", type=str, default=None, help="module:function path that returns normalized scenes")
    ap.add_argument("--suite-root", type=str, default=None, help="directory of normalized scene JSON/YAML files")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--ask-thresh", type=float, default=0.7)
    args = ap.parse_args()

    if not args.suite_loader and not args.suite_root:
        raise SystemExit("Provide either --suite-loader or --suite-root")

    scenes_raw = load_suites_from_loader(args.suite_loader) if args.suite_loader else load_suites_from_dir(args.suite_root)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or f"logs/rg_paper_{ts}"
    os.makedirs(outdir, exist_ok=True)

    kd = KnowDanger(Config(alpha=args.alpha, ask_threshold_confidence=args.ask_thresh))

    # CSV
    plan_csv_path = os.path.join(outdir, "plan_summary.csv")
    step_csv_path = os.path.join(outdir, "step_log.csv")
    with open(plan_csv_path, "w", newline="") as plan_csv, open(step_csv_path, "w", newline="") as step_csv:
        plan_writer = csv.writer(plan_csv); step_writer = csv.writer(step_csv)
        plan_writer.writerow(["variant","scene","plan","final_label","why"])
        step_writer.writerow(["variant","scene","plan","step_idx","action","rg_label","kn_label","final_label","rg_why","kn_why"])

        for scn in scenes_raw:
            scene = to_scene(scn)
            plans = [to_plan(p) for p in scn["plans"]]

            # RoboGuard only
            for p in plans:
                pa = kd.rg.evaluate_plan(p, kd.rg.compile_specs(scene))
                # wrap into rows for logging consistency
                # we mark final = RG label AND no KnowNo
                # For faithful RG paper evaluation, prefer to use RG repo's own runner; this is an overlay logger.
                # Here we just log RG step labels.
                for idx, v in enumerate(pa):
                    step_writer.writerow(["RoboGuard", scene.name, p.name, idx, p.steps[idx].action, v.label, None, v.label, v.why, ""])
                fin = "UNSAFE" if any(v.label=="UNSAFE" for v in pa) else ("UNCERTAIN" if any(v.label=="UNCERTAIN" for v in pa) else "SAFE")
                plan_writer.writerow(["RoboGuard", scene.name, p.name, fin, "RoboGuard-only"])

            # KnowDanger fusion (optional overlay)
            for p in plans:
                res = kd.run(scene, p)
                plan_writer.writerow(["KnowDanger", scene.name, p.name, res.overall.label, res.overall.why])
                for i, s in enumerate(res.steps):
                    step_writer.writerow(["KnowDanger", scene.name, p.name, i, s.step.action, s.roboguard.label, (s.knowno.label if s.knowno else None), s.final.label, s.roboguard.why, (s.knowno.why if s.knowno else "")])

    print(json.dumps({
        "outdir": outdir,
        "files": ["plan_summary.csv","step_log.csv"]
    }, indent=2))

if __name__ == "__main__":
    main()
