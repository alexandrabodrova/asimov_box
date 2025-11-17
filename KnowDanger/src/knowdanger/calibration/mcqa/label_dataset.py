
# label_dataset.py
# Interactive labeler: reads JSONL from build_dataset.py and writes JSONL with "safe_set".
#
# Usage:
#   python label_dataset.py --in /path/raw.jsonl --out /path/labeled.jsonl
#   python label_dataset.py --in /path/raw.jsonl --out /path/labeled.jsonl --resume
#
import argparse, json, os
from typing import List

def pretty_options(options: List[str], scores):
    lines = []
    for i, opt in enumerate(options):
        if scores is None:
            lines.append(f"  [{i}] {opt}")
        else:
            lines.append(f"  [{i}] {opt}    (score={scores[i]:.3f})")
    return "\n".join(lines)

def parse_indices(s: str, n: int) -> List[int]:
    s = s.strip()
    if s == "":
        return []
    out = set()
    for tok in s.replace(",", " ").split():
        j = int(tok)
        if j < 0 or j >= n:
            raise ValueError(f"Index {j} out of range [0, {n-1}]")
        out.add(j)
    return sorted(out)

def load_existing_labels(path: str):
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if "safe_set" in r:
                    labels[r["id"]] = r["safe_set"]
            except Exception:
                pass
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    labeled_ids = load_existing_labels(args.out) if args.resume else {}

    with open(args.out, "a" if args.resume else "w", encoding="utf-8") as fout:
        with open(args.inp, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if args.resume and row["id"] in labeled_ids:
                    continue

                print("="*90)
                print(f"ID: {row['id']}")
                print("Context:")
                print(row["context"])
                print("\nOptions:")
                print(pretty_options(row["options"], row.get("scores")))
                print("\nEnter ALL SAFE indices (space/comma). ENTER for none; 's' to skip; 'q' to quit.")

                while True:
                    x = input("> ").strip().lower()
                    if x == "q":
                        print("Stopping (partial labels saved).")
                        return
                    if x == "s":
                        row["safe_set"] = row.get("safe_set", [])
                        break
                    try:
                        row["safe_set"] = parse_indices(x, len(row["options"]))
                        break
                    except Exception as e:
                        print("Error:", e)

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. Wrote labels to {args.out}")

if __name__ == "__main__":
    main()
