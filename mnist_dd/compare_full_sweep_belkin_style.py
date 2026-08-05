"""Recreate Belkin's Fig. 3 layout (top: zero-one loss, bottom: squared/MSE loss)
but overlay SEVERAL full-sweep summaries against each other, for comparing
results from different sweep configurations (e.g. before/after a
sweep_config.py change) -- rather than plot_full_sweep_belkin_style.py's
one-summary-vs-Belkin comparison. Repeat --summary_path/--sweep_name once per
sweep; colors come from SWEEP_COLORS in the order given, the same order
plot_norm_curve.py uses, so a sweep keeps its color across both figures.

Reads the summary CSVs (as produced by aggregate_full_sweep.py, from anywhere
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
from full_sweep import RESULTS_ROOT
from utils import num_params

BELKIN_DIGITIZED_PATH = os.path.join(RESULTS_ROOT, "belkin_digitized.csv")
OUT_PATH = os.path.join(RESULTS_ROOT, "full_sweep_compare_belkin_style.png")

INTERPOLATION_THRESHOLD = K * N_TRAIN / 1e3  # params where num_params(H) == K*N_TRAIN

# Okabe-Ito, assigned to sweeps in this fixed order and never cycled, so a given
# sweep keeps its color across every figure. Checked with the dataviz validator:
# worst adjacent CVD dE 11.0, normal-vision 25.8, all >= 3:1 on white.
SWEEP_COLORS = ["#0072B2", "#D55E00", "#009E73"]  # blue, vermillion, bluish green
# Paired with the colors, because these arms coincide exactly over part of the
# H range -- a solid line drawn later would completely hide an earlier one there.
SWEEP_LINESTYLES = ["-", "--", ":"]   # default for plot_norm_curve.py
TEST_LINESTYLE, TRAIN_LINESTYLE = "-", "--"
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



def taper(i, n, hi, lo):
    """Value for series i of n, decreasing hi -> lo in draw order. These arms
    coincide exactly over parts of the H range, so later curves are drawn
    thinner and sit visibly on top of earlier ones."""
    return hi if n < 2 else hi + (lo - hi) * i / (n - 1)


def plot_sweep(ax_top, ax_bot, row, h_vals, color, label, lw):
    x = [num_params(h) / 1e3 for h in h_vals]
    ax_top.plot(x, [row[f"H{h}_test_zeroone_mean"] * 100 for h in h_vals],
                marker="D", ms=4, lw=lw, color=color, linestyle=TEST_LINESTYLE, label=f"{label} test")
    ax_top.plot(x, [row[f"H{h}_train_zeroone_mean"] * 100 for h in h_vals],
                lw=lw, color=color, linestyle=TRAIN_LINESTYLE, label=f"{label} train")
    # Our MSE is (outputs - y_onehot)**2 averaged over both N and K, but Belkin's
    # squared loss sums over the K one-hot outputs per example (only averaging
    # over N) -- so ours is 1/K of his units. Rescale by K to match (see
    # plot_full_sweep_belkin_style.py, same convention here).
    ax_bot.plot(x, [row[f"H{h}_test_MSE_mean"] * K for h in h_vals],
                marker="D", ms=4, lw=lw, color=color, linestyle=TEST_LINESTYLE, label=f"{label} test")
    ax_bot.plot(x, [row[f"H{h}_train_MSE_mean"] * K for h in h_vals],
                lw=lw, color=color, linestyle=TRAIN_LINESTYLE, label=f"{label} train")


def parse_linestyle(s):
    """CLI linestyle: matplotlib's own ("-", "--", ":") or a dash pattern
    ("6,1.5" -> (0, (6, 1.5))), which the named styles can't express."""
    if "," in s:
        return (0, tuple(float(x) for x in s.split(",")))
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_path", action="append", required=True, metavar="PATH",
                         help="Summary CSV to plot. Repeat once per sweep, up to 3.")
    parser.add_argument("--sweep_name", action="append", required=True, metavar="NAME",
                         help="Legend label for the corresponding --summary_path.")
    parser.add_argument("--lr", type=float, default=0.0005, help="lr row to pull from each summary (default: 0.0005).")
    parser.add_argument("--batch_size", type=int, default=32,
                         help="batch_size row to pull from each summary (default: 32; use 4000 for --full_batch runs).")
    parser.add_argument("--color", action="append", metavar="HEX",
                         help="Color per sweep, in order. Defaults to SWEEP_COLORS.")
    parser.add_argument("--ylim_zeroone", type=float, default=None,
                         help="Fix the top panel's upper limit. Use when two figures are meant "
                              "to be compared side by side, so the axes don't rescale between them.")
    parser.add_argument("--ylim_squared", type=float, default=None,
                         help="Fix the bottom panel's upper limit (see --ylim_zeroone).")
    parser.add_argument("--include_belkin", type=str2bool, default=False,
                         help="Also overlay Belkin's own digitized curves (black test / red train, dashed, "
                              "stars on test). Default: False.")
    parser.add_argument("--out_path", default=OUT_PATH, help="Path to save the plot to.")
    parser.add_argument("--plot_title", default=None,
                         help="Plot title (default: '{sweep_name1} vs {sweep_name2}', "
                              "plus ' vs Belkin Fig. 3' if --include_belkin=True).")
    args = parser.parse_args()

    if len(args.summary_path) != len(args.sweep_name):
        raise SystemExit(f"got {len(args.summary_path)} --summary_path but "
                          f"{len(args.sweep_name)} --sweep_name -- pass one name per path.")
    colors = args.color or SWEEP_COLORS
    if len(args.summary_path) > len(colors):
        raise SystemExit(f"{len(args.summary_path)} summaries but only {len(colors)} colors "
                          f"-- pass --color for each.")
    loaded = [load_row(p, args.lr, args.batch_size) for p in args.summary_path]

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

    n = len(loaded)
    for i, ((row, h_vals), color, name) in enumerate(zip(loaded, colors, args.sweep_name)):
        plot_sweep(ax_top, ax_bot, row, h_vals, color, name, taper(i, n, 2.6, 1.3))

    ax_top.axvline(INTERPOLATION_THRESHOLD, color="black", linestyle=":", alpha=0.5)
    ax_top.set_ylim(0, args.ylim_zeroone)
    ax_top.set_ylabel("Zero-one loss (%)")
    ax_top.legend(fontsize=7, ncol=2)

    ax_bot.axvline(INTERPOLATION_THRESHOLD, color="black", linestyle=":", alpha=0.5)
    ax_bot.set_ylim(0, args.ylim_squared)
    ax_bot.set_ylabel("Squared loss")
    ax_bot.set_xlabel(r"Number of parameters/weights ($\times10^3$)")
    ax_bot.legend(fontsize=7, ncol=2)

    if args.plot_title is None:
        title = " vs ".join(args.sweep_name)
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
