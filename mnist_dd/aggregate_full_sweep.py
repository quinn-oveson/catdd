"""Combine per-task full-sweep CSVs into one summary table.

Reads every results/full_sweep/task*.csv written by full_sweep.py and pivots
to one row per (lr, batch_size), with the mean and std (across seeds) of all
4 error curves -- train/test zero-one loss and train/test squared loss (MSE)
-- at each H in config.H_VALS, matching Belkin's Fig. 3 (which plots both
Test and Train for both loss types). overlay_belkin_figure.py only needs the
test_* columns, but the train_* columns are here too for a full 4-curve plot.

Also aggregates n_epochs (how many epochs each model actually ran before
hitting MAX_EPOCHS or early stopping -- see sweep_config.STOP_UNDERPARAM/
STOP_OVERPARAM) at every H, useful for checking where early stopping is
actually kicking in. Task CSVs always carry train_CE/test_CE too (evaluate()
computes all three losses regardless of which one was trained against), but
those columns are only pivoted into the summary when sweep_config.LOSS_FUNC
is CrossEntropyLoss, to keep MSE-run summaries free of an irrelevant metric.
"""
import glob
import os

import pandas as pd
import torch.nn as nn

from config import H_VALS
from full_sweep import RESULTS_DIR
from sweep_config import LOSS_FUNC

SUMMARY_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "full_sweep_summary.csv")


def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "task*.csv")))
    if not files:
        raise SystemExit(f"No task CSVs found in {RESULTS_DIR} -- run run_full_sweep_local.py first.")

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    metrics = ["test_zeroone", "test_MSE", "train_zeroone", "train_MSE", "n_epochs"]
    if isinstance(LOSS_FUNC, nn.CrossEntropyLoss):
        metrics += ["test_CE", "train_CE"]

    pivots = []
    for metric in metrics:
        pivot = df.pivot_table(index=["lr", "batch_size"], columns="H", values=metric, aggfunc=["mean", "std"])
        pivot.columns = [f"H{h}_{metric}_{stat}" for stat, h in pivot.columns]
        pivots.append(pivot)

    summary = pd.concat(pivots, axis=1).reset_index().sort_values(["batch_size", "lr"])

    missing_H = [h for h in H_VALS if f"H{h}_test_zeroone_mean" not in summary.columns]
    if missing_H:
        print(f"WARNING: no data found for H values {missing_H} -- check for missing/failed task CSVs.")

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.6f")
    print(f"Wrote {SUMMARY_PATH} ({len(summary)} grid cells from {len(df)} rows across {len(files)} task files)")


if __name__ == "__main__":
    main()
