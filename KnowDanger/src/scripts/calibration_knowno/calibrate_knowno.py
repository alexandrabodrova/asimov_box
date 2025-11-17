
"""
Calibrate KnowNo adapter (optional). You can supply a JSON file containing a list
of calibration score lists (each entry = list of option scores for one MCQA instance).
Usage:
    python scripts/calibrate_knowno.py --scores path/to/calibration.json
It prints the chosen tau and writes it into config/knowno.yaml (if present).
"""
import argparse, json
from pathlib import Path
from knowdanger.core.knowdanger_core import KnowDanger, Config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=str, required=True, help="Path to JSON: list of list[float]")
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    kd = KnowDanger(Config(alpha=args.alpha))
    with open(args.scores, "r") as f:
        score_sets = json.load(f)
    tau = kd.kn.calibrate(score_sets)
    print(json.dumps({"alpha": args.alpha, "tau": tau}, indent=2))

    # Persist to config if available
    cfg_path = Path("config/knowno.yaml")
    if cfg_path.exists():
        txt = cfg_path.read_text()
        txt += f"\n# Calibrated at runtime\ncalibrated_tau: {tau}\n"
        cfg_path.write_text(txt)

if __name__ == "__main__":
    main()
