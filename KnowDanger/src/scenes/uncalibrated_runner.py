
"""
uncalibrated_runner.py
----------------------
A simple runner that:
  1) Loads MCQA examples from a scene module (expects get_mcqa_examples()).
  2) Computes safety scores via score_stub.score_all (replace with your scorer).
  3) Picks the argmax-scoring option (UNCALIBRATED baseline).
  4) Logs full per-example records to run.jsonl and a summary.csv.

Example:
    python /mnt/data/knowdanger_cp/uncalibrated_runner.py \
        --module /path/to/example2_breakroom.py \
        --outdir /mnt/data/runs/breakroom_uncalib

If you ALSO have ground-truth safe_set for evaluation and want 'success' filled,
add --eval-with-labels PATH_TO_JSONL_LABELED (from label_cli.py).
"""

import argparse, json, os
from adapters.examples_loader import load_examples
from score_stub import score_all
from results_logger import ResultsLogger


def load_safe_labels(jsonl_path: str) -> dict:
    """
    Returns a mapping example_id -> safe_set indices from a labeled JSONL.
    """
    labels = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            labels[rec["id"]] = rec.get("safe_set", [])
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="Path to scenes module (.py) with get_mcqa_examples().")
    ap.add_argument("--outdir", required=True, help="Directory to store logs (JSONL + CSV).")
    ap.add_argument("--eval-with-labels", help="Optional: labeled JSONL from label_cli.py to compute 'success'.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data = load_examples(args.module)
    labels = load_safe_labels(args.eval_with_labels) if args.eval_with_labels else {}

    logger = ResultsLogger(args.outdir)

    for ex in data:
        ex_id = ex["id"]
        ctx = ex["context"]
        options = ex["options"]
        family = ex.get("meta", {}).get("family", "unknown")

        scores = score_all(ctx, options)
        chosen_idx = max(range(len(options)), key=lambda j: scores[j]) if options else None
        chosen_action = options[chosen_idx] if chosen_idx is not None else None

        # If we have labels, mark success iff chosen action is in the safe set
        safe_set = labels.get(ex_id, [])
        success = (chosen_idx in safe_set) if safe_set else None

        logger.log(
            example_id=ex_id,
            family=family,
            context=ctx,
            options=options,
            scores=scores,
            chosen_idx=chosen_idx,
            chosen_action=chosen_action,
            decision="UNCALIBRATED_ARGMAX",
            success=success,
            extra={"scores_min": min(scores) if scores else None,
                   "scores_max": max(scores) if scores else None,
                   "scores_mean": (sum(scores)/len(scores)) if scores else None}
        )

    logger.write_csv_summary()
    logger.close()
    print(f"Done. JSONL: {logger.jsonl_path}\nCSV: {logger.csv_path}")


if __name__ == "__main__":
    main()
