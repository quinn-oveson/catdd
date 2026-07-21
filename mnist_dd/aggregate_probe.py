"""Combine per-task probe CSVs into one summary table for manual review.

Reads every results/probe/task*.csv written by probe.py and pivots to one row
per (lr, batch_size), with the mean and std of test zero-one loss at each
probe H, averaged across seeds -- meant to be scanned by eye against Belkin's
approximate small-H targets, not filtered automatically.
"""
import glob
import os

import pandas as pd

from probe import PROBE_H_VALS, RESULTS_DIR

SUMMARY_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "probe_summary.csv")


def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "task*.csv")))
    if not files:
        raise SystemExit(f"No task CSVs found in {RESULTS_DIR} -- run run_stage1_local.py first.")

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    summary = df.pivot_table(
        index=["lr", "batch_size"],
        columns="H",
        values="test_zeroone",
        aggfunc=["mean", "std"],
    )
    summary.columns = [f"H{h}_test_zeroone_{stat}" for stat, h in summary.columns]
    summary = summary.reset_index().sort_values(["batch_size", "lr"])

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.4f")
    print(f"Wrote {SUMMARY_PATH} ({len(summary)} grid cells from {len(df)} rows across {len(files)} task files)")
    print(f"Belkin targets for reference: H={PROBE_H_VALS[0]}~55%, H={PROBE_H_VALS[1]}~48%, H={PROBE_H_VALS[2]}~25%")


if __name__ == "__main__":
    main()
