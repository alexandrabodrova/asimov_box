#!/usr/bin/env python3
"""
Summarize RoboGuard/KnowDanger evaluation CSVs.

Expected input schema per row:
  system, case_id, type, prompt, allowed, asr, notes

- 'type' is 'attack' or 'benign'
- 'allowed' is 0/1 (any action was allowed/executed)
- 'asr' is 0/1 (attack success: a harmful action executed)

Outputs:
  • CSV table with per-system metrics
  • A bar plot comparing Attack Success Rate (ASR) and Benign Block Rate (BBR)
"""

import argparse, glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["type"] = df["type"].str.lower()
    df["allowed"] = df["allowed"].astype(int)
    df["asr"] = df["asr"].astype(int)

    # Per-system tallies
    rows = []
    for system, g in df.groupby("system"):
        g_att = g[g["type"] == "attack"]
        g_ben = g[g["type"] == "benign"]

        total = len(g)
        n_att = len(g_att)
        n_ben = len(g_ben)

        asr = (g_att["asr"].sum() / n_att) if n_att > 0 else np.nan
        allowed_rate_attack = (g_att["allowed"].mean()) if n_att > 0 else np.nan

        # Benign block rate = 1 - allowed on benign
        benign_block_rate = (1.0 - g_ben["allowed"].mean()) if n_ben > 0 else np.nan
        allowed_rate_benign = (g_ben["allowed"].mean()) if n_ben > 0 else np.nan

        rows.append({
            "system": system,
            "total": total,
            "n_attack": n_att,
            "n_benign": n_ben,
            "ASR": asr,
            "allowed_rate_attack": allowed_rate_attack,
            "benign_block_rate": benign_block_rate,
            "allowed_rate_benign": allowed_rate_benign,
        })

    out = pd.DataFrame(rows).sort_values("system").reset_index(drop=True)
    return out

def plot_asr_bbr(summary_df: pd.DataFrame, out_path: str) -> None:
    systems = summary_df["system"].tolist()
    asr = summary_df["ASR"].values
    bbr = summary_df["benign_block_rate"].values

    x = np.arange(len(systems))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, asr, width, label="Attack Success Rate")
    plt.bar(x + width/2, bbr, width, label="Benign Block Rate")
    plt.xticks(x, systems, rotation=15)
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.title("ASR and Benign Block Rate by System")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, default="roboguard_eval_*.csv",
                    help="Glob of per-case CSVs to summarize")
    ap.add_argument("--out-table", type=str, default="roboguard_behavior_summary.csv",
                    help="Output CSV with per-system metrics")
    ap.add_argument("--out-plot", type=str, default="roboguard_behavior_asr_plot.png",
                    help="Output PNG comparing ASR and Benign Block Rate")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"[summarize] No files matched glob: {args.glob}")
        return 1

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"[summarize] Skipping {f}: {e}")
    if not dfs:
        print("[summarize] No readable CSVs.")
        return 1

    df = pd.concat(dfs, ignore_index=True)
    required = {"system", "case_id", "type", "prompt", "allowed", "asr"}
    missing = required - set(df.columns)
    if missing:
        print(f"[summarize] Missing columns in input CSVs: {missing}")
        return 1

    summary_df = summarize(df)
    summary_df.to_csv(args.out_table, index=False)
    print(f"[summarize] Wrote summary table: {args.out_table}")

    try:
        plot_asr_bbr(summary_df, args.out_plot)
        print(f"[summarize] Wrote plot: {args.out_plot}")
    except Exception as e:
        print(f"[summarize] Plotting failed (continuing): {e}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
