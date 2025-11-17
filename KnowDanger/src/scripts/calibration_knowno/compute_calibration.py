
"""
Compute the conformal safety threshold from a labeled JSONL (calibration set).

We implement a conservative split-conformal quantile. If SciPy is available,
we also offer an optional dataset-conditional Beta inversion to choose eps_hat.
"""

import argparse, json, math, sys, statistics
from typing import List, Dict, Any, Tuple

try:
    from scipy.stats import beta as sp_beta  # optional
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def quantile_conservative(xs: List[float], q: float) -> float:
    """
    Conservative one-sided empirical quantile with 'higher' method.
    """
    if not xs:
        raise ValueError("Empty list of values.")
    xs_sorted = sorted(xs)
    # index per 'higher' quantile
    k = math.ceil(q * len(xs_sorted)) - 1
    k = max(0, min(k, len(xs_sorted)-1))
    return xs_sorted[k]


def choose_eps_hat_dataset_conditional(N: int, delta: float, target_coverage: float) -> float:
    """
    If SciPy is present, binary search eps_hat s.t.
    Beta^{-1}_{N+1-v, v}(delta) ~= target_coverage, v=floor((N+1)*eps_hat).
    Fallback: return 1 - target_coverage (i.e., split-conformal level).
    """
    if not SCIPY_OK:
        return 1.0 - target_coverage

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        v = math.floor((N + 1) * mid)
        a, b = N + 1 - v, v
        if a <= 0 or b <= 0:
            # Edge cases: push inside (0,1)
            if v == 0:
                lo = mid
                continue
            if v == N + 1:
                hi = mid
                continue
        cov = sp_beta.ppf(delta, a, b)  # returns quantile
        if cov >= target_coverage:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def calibrate_from_jsonl(path: str, target_coverage: float, delta: float) -> Dict[str, Any]:
    chosen_scores: List[float] = []
    N_raw = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            N_raw += 1
            rec = json.loads(line)
            options = rec["options"]
            safe = rec["safe_set"]
            scores = rec.get("scores", None)

            if not safe:
                # scenes with no safe action are skipped for β-mapping
                continue
            if scores is None:
                # If no scores present, we cannot calibrate; user should re-run labeling with --with-scores
                raise ValueError("Record has no 'scores'. Re-run label_cli.py with --with-scores.")

            # β(x,Y) = argmax_{y∈Y} f(x,y)
            best_idx = max(safe, key=lambda j: scores[j])
            chosen_scores.append(scores[best_idx])

    if not chosen_scores:
        raise ValueError("No calibratable examples (did you mark any safe options and include scores?).")

    kappa = [1.0 - s for s in chosen_scores]  # non-conformity
    N = len(kappa)

    eps_hat = choose_eps_hat_dataset_conditional(N=N, delta=delta, target_coverage=target_coverage)

    # empirical quantile at level q_level = (N+1)*(1-eps_hat)/N
    q_level = ((N + 1) * (1.0 - eps_hat)) / N
    q_hat = quantile_conservative(kappa, q_level)

    return {
        "q_hat": float(q_hat),
        "eps_hat": float(eps_hat),
        "N_effective": int(N),
        "N_labeled": int(N_raw),
        "delta": float(delta),
        "target_coverage": float(target_coverage),
        "note": "If SCIPY_OK is false, eps_hat was set to 1 - target_coverage (split-conformal)."
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True, help="Path to labeled JSONL from label_cli.py")
    ap.add_argument("--out", required=True, help="Output JSON artifact path")
    ap.add_argument("--target-coverage", type=float, default=0.9, help="Desired coverage, e.g., 0.9 for 90%%")
    ap.add_argument("--delta", type=float, default=0.01, help="Dataset-conditional delta (if SciPy available)")
    args = ap.parse_args()

    art = calibrate_from_jsonl(args.calib, args.target_coverage, args.delta)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(art, f, indent=2)
    print(json.dumps(art, indent=2))
    print(f"\nWrote calibration artifact to {args.out}")


if __name__ == "__main__":
    main()
