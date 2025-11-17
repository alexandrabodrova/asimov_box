
"""
Interactive labeling CLI:
- Loads examples via adapters/examples_loader.py from a Python module path.
- Optionally computes safety scores via score_stub.py.
- Lets you mark which options are SAFE for each example (one or more).
- Writes a JSONL calibration file with fields:
  {id, context, options, safe_set, scores, meta}
"""

import argparse, json, sys
from typing import List, Dict, Any
from schemas import Example
from adapters.examples_loader import load_examples
from score_stub import score_all


def pretty_opts(options: List[str], scores: List[float] | None) -> str:
    lines = []
    for j, opt in enumerate(options):
        if scores is None:
            lines.append(f"  [{j}] {opt}")
        else:
            lines.append(f"  [{j}] {opt}    (score={scores[j]:.3f})")
    return "\n".join(lines)


def parse_indices(inp: str, n: int) -> List[int]:
    inp = inp.strip()
    if inp == "":
        return []
    out = []
    for tok in inp.replace(",", " ").split():
        j = int(tok)
        if j < 0 or j >= n:
            raise ValueError(f"Index {j} out of range [0, {n-1}]")
        out.append(j)
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="Path to scenes module (.py) with get_mcqa_examples().")
    ap.add_argument("--out", required=True, help="Output JSONL calibration file.")
    ap.add_argument("--with-scores", action="store_true", help="Compute toy safety scores via score_stub.py.")
    args = ap.parse_args()

    raw = load_examples(args.module)

    with open(args.out, "w", encoding="utf-8") as fout:
        for ex in raw:
            ex_id = ex.get("id")
            ctx = ex.get("context", "")
            options = ex.get("options", [])
            meta = ex.get("meta", {})

            print("="*80)
            print(f"ID: {ex_id}")
            print("Context:")
            print(ctx)
            scores = score_all(ctx, options) if args.with_scores else None
            print("\nOptions:")
            print(pretty_opts(options, scores))

            while True:
                try:
                    s = input("\nEnter SAFE option indices (space/comma-separated), or ENTER if none are safe: ")
                    safe_set = parse_indices(s, len(options))
                    break
                except Exception as e:
                    print("Error:", e)

            record = {
                "id": ex_id,
                "context": ctx,
                "options": options,
                "safe_set": safe_set,
                "scores": scores,
                "meta": meta,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"→ Saved {ex_id} with safe_set={safe_set}")

    print(f"\nDone. Wrote {args.out}")


if __name__ == "__main__":
    main()
