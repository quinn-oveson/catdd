"""Recreate Belkin's Fig. 3 layout (top: zero-one loss, bottom: squared/MSE loss)
but overlay TWO full-sweep summaries against each other, for comparing results
from two different sweep configurations (e.g. before/after a sweep_config.py
change) -- rather than plot_full_sweep_belkin_style.py's one-summary-vs-Belkin
comparison.

Reads two summary CSVs (as produced by aggregate_full_sweep.py, from anywhere
on disk -- e.g. a renamed copy scp'd down from a prior sweep alongside the
current results/full_sweep_summary.csv) plus optionally
results/belkin_digitized.csv. Each sweep gets its own color, used for both its
test and train curves on both plots (test has a 'D' marker, train doesn't --
same convention as plot_full_sweep_belkin_style.py, just recolored by sweep
instead of by test/train). Belkin's own curves (both black, dashed, same 'D'
marker on test) are only added with --include_belkin=True.
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from config import K, N_TRAIN
from full_sweep import RESULTS_DIR
from utils import num_params

BELKIN_DIGITIZED_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "belkin_digitized.csv")
OUT_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "full_sweep_compare_belkin_style.png")

INTERPOLATION_THRESHOLD = K * N_TRAIN / 1e3  # params where num_params(H) == K*N_TRAIN

# Okabe-Ito colorblind-safe pair (same palette family as plot_full_sweep_belkin_style.py's
# blue/orange, but purple/green instead so this plot's sweep colors don't read as
# "test vs. train" the way blue/orange does there) -- also distinct from
# BELKIN_COLOR below so a --include_belkin=True plot never has two series
# sharing a color.
SWEEP1_COLOR = "#CC79A7"  # reddish purple
SWEEP2_COLOR = "#009E73"  # bluish green
BELKIN_COLOR = "black"  # both test and train -- distinguished by marker (test='D', same as the sweeps) not color


def str2bool(v):
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected a bool (True/False), got {v!r}")


def h_vals_in(summary):
    return sorted(int(c[len("H"):-len("_test_zeroone_mean")])
                  for c in summary.columns if c.startswith("H") and c.endswith("_test_zeroone_mean"))


def load_row(summary_path, lr, batch_size):
    if not os.path.exists(summary_path):
        raise SystemExit(f"{summary_path} not found -- run aggregate_full_sweep.py first.")
    summary = pd.read_csv(summary_path)
    row_match = summary[(summary["lr"] == lr) & (summary["batch_size"] == batch_size)]
    if row_match.empty:
        raise SystemExit(f"No row with lr={lr}, batch_size={batch_size} in {summary_path}.")
    return row_match.iloc[0], h_vals_in(summary)


def plot_sweep(ax_top, ax_bot, row, h_vals, color, label):
    x = [num_params(h) / 1e3 for h in h_vals]
    ax_top.plot(x, [row[f"H{h}_test_zeroone_mean"] * 100 for h in h_vals],
                marker="D", ms=4, color=color, label=f"{label} test")
    ax_top.plot(x, [row[f"H{h}_train_zeroone_mean"] * 100 for h in h_vals],
                color=color, label=f"{label} train")
    # Our MSE is (outputs - y_onehot)**2 averaged over both N and K, but Belkin's
    # squared loss sums over the K one-hot outputs per example (only averaging
    # over N) -- so ours is 1/K of his units. Rescale by K to match (see
    # plot_full_sweep_belkin_style.py, same convention here).
    ax_bot.plot(x, [row[f"H{h}_test_MSE_mean"] * K for h in h_vals],
                marker="D", ms=4, color=color, label=f"{label} test")
    ax_bot.plot(x, [row[f"H{h}_train_MSE_mean"] * K for h in h_vals],
                color=color, label=f"{label} train")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_path1", required=True, help="First summary CSV to plot.")
    parser.add_argument("--summary_path2", required=True, help="Second summary CSV to plot.")
    parser.add_argument("--sweep_name1", required=True, help="Legend label for the first summary.")
    parser.add_argument("--sweep_name2", required=True, help="Legend label for the second summary.")
    parser.add_argument("--lr1", type=float, default=0.0005, help="lr row to pull from summary_path1 (default: 0.0005).")
    parser.add_argument("--batch_size1", type=int, default=32, help="batch_size row to pull from summary_path1 (default: 32).")
    parser.add_argument("--lr2", type=float, default=0.0005, help="lr row to pull from summary_path2 (default: 0.0005).")
    parser.add_argument("--batch_size2", type=int, default=32, help="batch_size row to pull from summary_path2 (default: 32).")
    parser.add_argument("--include_belkin", type=str2bool, default=False,
                         help="Also overlay Belkin's own digitized curves (black test / red train, dashed, "
                              "stars on test). Default: False.")
    parser.add_argument("--out_path", default=OUT_PATH, help="Path to save the plot to.")
    parser.add_argument("--plot_title", default=None,
                         help="Plot title (default: '{sweep_name1} vs {sweep_name2}', "
                              "plus ' vs Belkin Fig. 3' if --include_belkin=True).")
    args = parser.parse_args()

    row1, h_vals1 = load_row(args.summary_path1, args.lr1, args.batch_size1)
    row2, h_vals2 = load_row(args.summary_path2, args.lr2, args.batch_size2)

    if args.include_belkin and not os.path.exists(BELKIN_DIGITIZED_PATH):
        raise SystemExit(f"{BELKIN_DIGITIZED_PATH} not found -- run extract_belkin_markers.py first.")

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if args.include_belkin:
        belkin = pd.read_csv(BELKIN_DIGITIZED_PATH).sort_values("H")
        x_belkin = belkin["N"] / 1e3
        ax_top.plot(x_belkin, belkin["test_zeroone_pct"], linestyle="--", marker="D", ms=4,
                    color=BELKIN_COLOR, label="Belkin test")
        ax_top.plot(x_belkin, belkin["train_zeroone_pct"], linestyle="--",
                    color=BELKIN_COLOR, label="Belkin train")
        ax_bot.plot(x_belkin, belkin["test_squared_loss"], linestyle="--", marker="D", ms=4,
                    color=BELKIN_COLOR, label="Belkin test")
        ax_bot.plot(x_belkin, belkin["train_squared_loss"], linestyle="--",
                    color=BELKIN_COLOR, label="Belkin train")

    plot_sweep(ax_top, ax_bot, row1, h_vals1, SWEEP1_COLOR, args.sweep_name1)
    plot_sweep(ax_top, ax_bot, row2, h_vals2, SWEEP2_COLOR, args.sweep_name2)

    ax_top.axvline(INTERPOLATION_THRESHOLD, color="black", linestyle=":", alpha=0.5)
    ax_top.set_ylim(bottom=0)
    ax_top.set_ylabel("Zero-one loss (%)")
    ax_top.legend(fontsize=7, ncol=2)

    ax_bot.axvline(INTERPOLATION_THRESHOLD, color="black", linestyle=":", alpha=0.5)
    ax_bot.set_ylim(bottom=0)
    ax_bot.set_ylabel("Squared loss")
    ax_bot.set_xlabel(r"Number of parameters/weights ($\times10^3$)")
    ax_bot.legend(fontsize=7, ncol=2)

    if args.plot_title is None:
        title = f"{args.sweep_name1} vs {args.sweep_name2}"
        if args.include_belkin:
            title += " vs Belkin Fig. 3"
        ax_top.set_title(title)
    else:
        ax_top.set_title(args.plot_title)

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
