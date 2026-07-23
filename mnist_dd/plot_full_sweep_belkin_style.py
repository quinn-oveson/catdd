"""Recreate Belkin's Fig. 3 layout (top: zero-one loss, bottom: squared/MSE loss,
each with test in blue and train in orange) and overlay it with one full-sweep
candidate's own test/train curves, for a direct visual double-descent comparison.

Reads results/belkin_digitized.csv (dashed lines) and results/full_sweep_summary.csv
(solid lines, filtered down to a single --lr/--batch_size row -- default
lr=0.0005, batch_size=32). Belkin is black (test)/red (train) and ours is
blue (test)/orange (train), plus Belkin's curves are dashed and ours are
solid with marker-D-on-test (from plot.py), so the two are easy to tell
apart at a glance.
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from config import K, N_TRAIN
from full_sweep import RESULTS_DIR
from utils import num_params

SUMMARY_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "full_sweep_summary.csv")
BELKIN_DIGITIZED_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "belkin_digitized.csv")
OUT_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "full_sweep_belkin_style.png")

INTERPOLATION_THRESHOLD = K * N_TRAIN / 1e3  # params where num_params(H) == K*N_TRAIN

TEST_COLOR = "tab:blue"
TRAIN_COLOR = "tab:orange"
BELKIN_TEST_COLOR = "black"
BELKIN_TRAIN_COLOR = "tab:red"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_path", default=SUMMARY_PATH,
                         help="Which full-sweep summary CSV to plot (default: results/full_sweep_summary.csv).")
    parser.add_argument("--out_path", default=OUT_PATH,
                         help="Path to save the plot to")
    parser.add_argument("--lr", type=float, default=0.0005,
                         help="lr row to pull from the summary (default: 0.0005).")
    parser.add_argument("--batch_size", type=int, default=32,
                         help="batch_size row to pull from the summary (default: 32).")
    parser.add_argument("--plot_title", default=None,
                        help="What title to give this plot (default: lr={args.lr}, batch_size={args.batch_size} vs. Belkin Fig. 3).")

    args = parser.parse_args()

    if not os.path.exists(args.summary_path):
        raise SystemExit(f"{args.summary_path} not found -- run aggregate_full_sweep.py first.")
    if not os.path.exists(BELKIN_DIGITIZED_PATH):
        raise SystemExit(f"{BELKIN_DIGITIZED_PATH} not found -- run extract_belkin_markers.py first.")

    summary = pd.read_csv(args.summary_path)
    belkin = pd.read_csv(BELKIN_DIGITIZED_PATH).sort_values("H")

    row_match = summary[(summary["lr"] == args.lr) & (summary["batch_size"] == args.batch_size)]
    if row_match.empty:
        raise SystemExit(f"No row with lr={args.lr}, batch_size={args.batch_size} in {args.summary_path}.")
    row = row_match.iloc[0]

    h_vals = sorted(int(c[len("H"):-len("_test_zeroone_mean")])
                     for c in summary.columns if c.startswith("H") and c.endswith("_test_zeroone_mean"))
    x_ours = [num_params(h) / 1e3 for h in h_vals]
    x_belkin = belkin["N"] / 1e3

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_top.plot(x_belkin, belkin["test_zeroone_pct"], linestyle="--", marker="*", ms=8,
                color=BELKIN_TEST_COLOR, label="Belkin test")
    ax_top.plot(x_belkin, belkin["train_zeroone_pct"], linestyle="--",
                color=BELKIN_TRAIN_COLOR, label="Belkin train")
    ax_top.plot(x_ours, [row[f"H{h}_test_zeroone_mean"] * 100 for h in h_vals],
                marker="D", ms=4, color=TEST_COLOR, label="Ours test")
    ax_top.plot(x_ours, [row[f"H{h}_train_zeroone_mean"] * 100 for h in h_vals],
                color=TRAIN_COLOR, label="Ours train")
    ax_top.axvline(INTERPOLATION_THRESHOLD, color="black", linestyle=":", alpha=0.5)
    ax_top.set_ylim(bottom=0)
    ax_top.set_ylabel("Zero-one loss (%)")
    if args.plot_title == None:
        ax_top.set_title(f"lr={args.lr}, batch_size={args.batch_size} vs. Belkin Fig. 3")
    else:
        ax_top.set_title(args.plot_title)
    ax_top.legend(fontsize=7, ncol=2)

    ax_bot.plot(x_belkin, belkin["test_squared_loss"], linestyle="--", marker="*", ms=8,
                color=BELKIN_TEST_COLOR, label="Belkin test")
    ax_bot.plot(x_belkin, belkin["train_squared_loss"], linestyle="--",
                color=BELKIN_TRAIN_COLOR, label="Belkin train")
    # Our MSE is (outputs - y_onehot)**2 averaged over both N and K, but Belkin's
    # squared loss sums over the K one-hot outputs per example (only averaging
    # over N) -- so ours is 1/K of his units. Rescale by K to match.
    ax_bot.plot(x_ours, [row[f"H{h}_test_MSE_mean"] * K for h in h_vals],
                marker="D", ms=4, color=TEST_COLOR, label="Ours test")
    ax_bot.plot(x_ours, [row[f"H{h}_train_MSE_mean"] * K for h in h_vals],
                color=TRAIN_COLOR, label="Ours train")
    ax_bot.axvline(INTERPOLATION_THRESHOLD, color="black", linestyle=":", alpha=0.5)
    ax_bot.set_ylim(bottom=0)
    ax_bot.set_ylabel("Squared loss")
    ax_bot.set_xlabel(r"Number of parameters/weights ($\times10^3$)")
    ax_bot.legend(fontsize=7, ncol=2)

    ticks = [3, 10, 40, 100, 300, 800]
    ax_bot.set_xscale("log")
    ax_bot.set_xticks(ticks)
    ax_bot.xaxis.set_major_formatter(ScalarFormatter())
    ax_bot.ticklabel_format(style="plain", axis="x")

    fig.tight_layout()
    fig.savefig(args.out_path, dpi=150)
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
