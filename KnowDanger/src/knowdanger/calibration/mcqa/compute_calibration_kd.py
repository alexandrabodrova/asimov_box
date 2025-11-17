
# compute_calibration.py
# Standalone: compute conformal q_hat from a labeled JSONL (β-mapping + dataset-conditional bound).
#
# Usage:
#   python compute_calibration.py \
#       --calib /path/labeled.jsonl \
#       --out   /path/calibration_artifact.json \
#       --target-coverage 0.90 \
#       --delta 0.01
#
# JSONL rows must contain:
#   - "scores": List[float]  (per-option scores in [0,1], higher = safer/correct)
#   - "safe_set": List[int]  (indices of acceptable/safe options; may be multiple)
#
# Output JSON:
#   {"q_hat": float, "eps_hat": float, "N_effective": int, "N_labeled": int,
#    "delta": float, "target_coverage": float, "note": str}
#
import argparse, json, math

try:
    from scipy.stats import beta as sp_beta
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

def conservative_quantile(xs, level):
    if not xs:
        raise ValueError("Empty list for quantile.")
    xs = sorted(xs)
    k = max(0, min(len(xs)-1, math.ceil(level*len(xs)) - 1))
    return float(xs[k])

def choose_eps_hat_dataset_conditional(N, delta, target_coverage):
    if not SCIPY_OK:
        return 1.0 - target_coverage
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5*(lo+hi)
        v = math.floor((N+1)*mid)
        a, b = N+1 - v, v
        if a <= 0 or b <= 0:
            if v == 0:  lo = mid; continue
            if v == N+1: hi = mid; continue
        cov = sp_beta.ppf(delta, a, b)
        if cov >= target_coverage: hi = mid
        else: lo = mid
    return 0.5*(lo+hi)

def calibrate_from_jsonl(path, target_coverage, delta):
    kappas = []
    N_raw = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            N_raw += 1
            r = json.loads(line)
            scores = r.get("scores")
            safe   = r.get("safe_set", [])
            if not scores or not safe:
                continue
            j_star = max(safe, key=lambda j: scores[j])  # β(x, Y): best acceptable
            s = float(scores[j_star])
            kappas.append(1.0 - s)
    if not kappas:
        raise ValueError("No rows with both 'scores' and non-empty 'safe_set'.")
    N = len(kappas)
    eps_hat = choose_eps_hat_dataset_conditional(N, delta, target_coverage)
    q_level = ((N + 1) * (1.0 - eps_hat)) / N
    q_hat = conservative_quantile(kappas, q_level)
    return {
        "q_hat": float(q_hat),
        "eps_hat": float(eps_hat),
        "N_effective": int(N),
        "N_labeled": int(N_raw),
        "delta": float(delta),
        "target_coverage": float(target_coverage),
        "note": ("SciPy used for dataset-conditional bound" if SCIPY_OK
                 else "SciPy not available: eps_hat := 1 - target_coverage (split CP).")
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True, help="Labeled JSONL (from your labeler)")
    ap.add_argument("--out", required=True, help="Output JSON artifact with q_hat")
    ap.add_argument("--target-coverage", type=float, default=0.90)
    ap.add_argument("--delta", type=float, default=0.01)
    args = ap.parse_args()
    art = calibrate_from_jsonl(args.calib, args.target_coverage, args.delta)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(art, f, indent=2)
    print(json.dumps(art, indent=2))
    print(f"\nWrote calibration artifact to {args.out}")

if __name__ == "__main__":
    main()
