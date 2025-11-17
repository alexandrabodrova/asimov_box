
"""
results_logger.py
-----------------
Drop-in utilities to save experiment/test runs (JSONL + optional CSV summary).

Usage 1 (drop-in from your own runner):
--------------------------------------
from results_logger import ResultsLogger

logger = ResultsLogger(out_dir="runs/2025-10-17_uncalibrated")
for ex in examples:
    # ... run your uncalibrated KnowNo/KnowDanger here ...
    logger.log(
        example_id=ex["id"],
        family=ex.get("meta",{}).get("family","unknown"),
        context=ex["context"],
        options=ex["options"],
        scores=scores,                  # list[float] or None
        chosen_idx=chosen_idx,          # int or None
        chosen_action=ex["options"][chosen_idx] if chosen_idx is not None else None,
        decision="UNCALIBRATED_ARGMAX", # or your policy tag
        success=bool(is_safe) if is_safe is not None else None,  # if you know ground-truth safety
        extra={"note": "any dict-like extras go here"}
    )
# writes JSONL incrementally; you can also ask for a CSV aggregate:
logger.write_csv_summary()  # optional
logger.close()


Usage 2 (one-shot helper if you don't want a class):
---------------------------------------------------
from results_logger import log_result_jsonl
log_result_jsonl(path="run.jsonl", record={...})

File formats
------------
- JSONL file with one record per example (full detail)
- CSV summary with common fields to make quick tables
"""

from __future__ import annotations
import os, json, csv, datetime
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class ResultsLogger:
    def __init__(self, out_dir: str):
        """
        Args:
            out_dir: directory where artifacts will be stored.
                     We'll create:
                        - out_dir/run.jsonl     (append-only, per-example records)
                        - out_dir/summary.csv   (optional aggregate)
                        - out_dir/meta.json     (run metadata)
        """
        _ensure_dir(out_dir)
        self.out_dir = out_dir
        self.jsonl_path = os.path.join(out_dir, "run.jsonl")
        self.csv_path = os.path.join(out_dir, "summary.csv")
        self.meta_path = os.path.join(out_dir, "meta.json")
        self._n = 0

        if not os.path.exists(self.meta_path):
            meta = {
                "created_at": _now_iso(),
                "out_dir": os.path.abspath(out_dir),
                "schema": {
                    "jsonl": [
                        "ts", "example_id", "family", "context", "options",
                        "scores", "chosen_idx", "chosen_action",
                        "decision", "success", "extra"
                    ]
                }
            }
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    def log(self,
            example_id: str,
            family: Optional[str],
            context: str,
            options: List[str],
            scores: Optional[List[float]],
            chosen_idx: Optional[int],
            chosen_action: Optional[str],
            decision: str,
            success: Optional[bool],
            extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Append one record to run.jsonl.
        """
        rec = {
            "ts": _now_iso(),
            "example_id": example_id,
            "family": family,
            "context": context,
            "options": options,
            "scores": scores,
            "chosen_idx": chosen_idx,
            "chosen_action": chosen_action,
            "decision": decision,
            "success": success,
            "extra": extra or {},
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._n += 1

    def write_csv_summary(self) -> None:
        """
        Convert the JSONL file into a flat CSV with common fields.
        """
        # read all
        rows = []
        if not os.path.exists(self.jsonl_path):
            return
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        # write csv
        fieldnames = [
            "ts", "example_id", "family",
            "decision", "success", "chosen_idx", "chosen_action"
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})

    def close(self) -> None:
        # nothing to close; method provided for symmetry with other loggers
        pass


def log_result_jsonl(path: str, record: Dict[str, Any]) -> None:
    """
    One-shot append to a JSONL file. Useful if you don't want the class.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
