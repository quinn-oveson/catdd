"""Combine per-task full-sweep CSVs into one summary table.

Reads every task*.csv written by full_sweep.py for the sweep preset named by
$CATDD_SWEEP (results/full_sweep/<preset>/, see sweep_config.py) and pivots to
one row per (lr, batch_size), with the mean and std (across seeds) of every
recorded per-H quantity, writing results/full_sweep_summary_<preset>.csv.

Aggregated at each H in config.H_VALS:
  - all 4 error curves -- train/test zero-one loss and train/test squared loss
    (MSE) -- matching Belkin's Fig. 3, which plots both Test and Train for both
    loss types. overlay_belkin_figure.py only needs the test_* columns, but the
    train_* columns are here too for a full 4-curve plot.
  - n_epochs: how many epochs each model actually ran before hitting MAX_EPOCHS
    or early stopping (see sweep_config.STOP_UNDERPARAM/STOP_OVERPARAM),
    useful for checking where early stopping is actually kicking in.
  - the weight norms (init_weight_norm, weight_norm, hidden_weight_norm,
    output_weight_norm) -- the norm of the parameter vector before and after
    training. Unlike the CE columns these are always aggregated regardless of
    LOSS_FUNC, since a weight norm means the same thing under either loss.

Task CSVs always carry train_CE/test_CE too (evaluate() computes all three
losses regardless of which one was trained against), but those columns are
only pivoted into the summary when sweep_config.LOSS_FUNC is CrossEntropyLoss,
to keep MSE-run summaries free of an irrelevant metric.

Any metric missing from the task CSVs entirely is skipped rather than raising,
so a results directory written before a metric existed still aggregates.
"""
import argparse
import glob
import os

import pandas as pd
import torch.nn as nn

from config import H_VALS
from full_sweep import RESULTS_DIR, RESULTS_ROOT
from sweep_config import SWEEP_NAME, LOSS_FUNC

SUMMARY_PATH = os.path.join(RESULTS_ROOT, f"full_sweep_summary_{SWEEP_NAME}.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default=RESULTS_DIR,
                         help="Directory of per-task CSVs to aggregate (default: the "
                              "current $CATDD_SWEEP preset's results/full_sweep/<preset>/).")
    parser.add_argument("--summary_path", default=SUMMARY_PATH,
                         help="Where to write the summary CSV (default: "
                              "results/full_sweep_summary_<preset>.csv).")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "task*.csv")))
    if not files:
        raise SystemExit(f"No task CSVs found in {args.results_dir} -- run run_full_sweep_local.py first.")

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    metrics = ["test_zeroone", "test_MSE", "train_zeroone", "train_MSE", "n_epochs"]
    if isinstance(LOSS_FUNC, nn.CrossEntropyLoss):
        metrics += ["test_CE", "train_CE"]
    metrics += ["init_weight_norm", "weight_norm", "hidden_weight_norm", "output_weight_norm"]

    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        print(f"WARNING: task CSVs in {args.results_dir} have no {missing_metrics} column(s) -- "
              f"skipping those (were they written before that metric was added?).")
        metrics = [m for m in metrics if m in df.columns]

    pivots = []
    for metric in metrics:
        pivot = df.pivot_table(index=["lr", "batch_size"], columns="H", values=metric, aggfunc=["mean", "std"])
        pivot.columns = [f"H{h}_{metric}_{stat}" for stat, h in pivot.columns]
        pivots.append(pivot)

    summary = pd.concat(pivots, axis=1).reset_index().sort_values(["batch_size", "lr"])

    missing_H = [h for h in H_VALS if f"H{h}_test_zeroone_mean" not in summary.columns]
    if missing_H:
        print(f"WARNING: no data found for H values {missing_H} -- check for missing/failed task CSVs.")

    os.makedirs(os.path.dirname(args.summary_path), exist_ok=True)
    summary.to_csv(args.summary_path, index=False, float_format="%.6f")
    print(f"Wrote {args.summary_path} ({len(summary)} grid cells from {len(df)} rows across {len(files)} task files)")


if __name__ == "__main__":
    main()
