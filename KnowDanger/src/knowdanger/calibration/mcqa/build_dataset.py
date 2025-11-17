
# build_dataset.py
# Build a combined MCQA dataset JSONL from scene modules.
#
# Usage:
#   python build_dataset.py --modules M1.py M2.py --out /path/out.jsonl
#   python build_dataset.py --modules M1.py --out out.jsonl --scorer score_knowno_like:score_all
#
import argparse, json, os, importlib
from typing import List, Dict, Any
from examples_loader_kd import load_examples_from_module

def import_scorer(s: str):
    if not s:
        return None
    if ":" in s:
        mod_name, fn_name = s.split(":", 1)
    else:
        mod_name, fn_name = s, "score_all"
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, fn_name):
        raise AttributeError(f"{mod_name} has no function {fn_name}")
    return getattr(mod, fn_name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-per-module", type=int, default=0)
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--scorer", default="")
    args = ap.parse_args()

    scorer_fn = import_scorer(args.scorer) if args.scorer else None

    seen = set()
    n_written = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        for mod_path in args.modules:
            data = load_examples_from_module(mod_path)
            if args.limit_per_module > 0:
                data = data[:args.limit_per_module]
            for ex in data:
                key = (ex["context"], tuple(ex["options"]))
                if args.dedupe and key in seen:
                    continue
                seen.add(key)
                row = {
                    "id": ex["id"],
                    "context": ex["context"],
                    "options": ex["options"],
                    "meta": ex.get("meta", {}),
                }
                if scorer_fn is not None:
                    row["scores"] = [float(s) for s in scorer_fn(ex["context"], ex["options"], ex.get("meta"))]
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
    print(f"Wrote {n_written} rows to {args.out}")

if __name__ == "__main__":
    main()
