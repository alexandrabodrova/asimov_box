
# kd_calibrate.py
# Config-driven wrapper for calibration (YAML or JSON config).
#
# Config keys:
#   labeled_jsonl: "/path/to/labeled.jsonl"
#   artifact_out:  "/path/to/out.json"
#   target_coverage: 0.90
#   delta: 0.01
#
# Usage:
#   python kd_calibrate.py --cfg /path/config.yaml
#
import argparse, json, os, sys
from typing import Any, Dict

# Reuse logic from compute_calibration.py (import if available)
try:
    from compute_calibration_kd import calibrate_from_jsonl
except Exception as e:
    print("ERROR: compute_calibration.calibrate_from_jsonl not found. Place this file next to compute_calibration.py.", file=sys.stderr)
    raise

def load_cfg(path: str) -> Dict[str, Any]:
    # Try YAML first, then JSON fallback
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="YAML/JSON config path")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    calib = cfg["labeled_jsonl"]
    out   = cfg["artifact_out"]
    target = float(cfg.get("target_coverage", 0.90))
    delta  = float(cfg.get("delta", 0.01))

    art = calibrate_from_jsonl(calib, target, delta)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(art, f, indent=2)
    print(json.dumps(art, indent=2))
    print(f"\nWrote calibration artifact to {out}")

if __name__ == "__main__":
    main()
