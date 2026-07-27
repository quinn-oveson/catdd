"""Overlay two full-sweep summaries as error-over-weight-norm, for testing
whether the double-descent spike at the interpolation threshold tracks weight
norm rather than overparameterization itself.

The intended pair is the `belkin` and `belkin_noreuse` presets (see
sweep_config.py): identical in every respect except that `belkin` reuses the
previous, smaller model's trained weights at every underparameterized H while
`belkin_noreuse` starts every H from a fresh Glorot init. Under `belkin` the
underparameterized models therefore inherit a norm that has been growing along
the whole chain, and the first overparameterized H (whose own reuse flag is
False in Belkin's setup) drops back to Glorot scale -- a discontinuity at
exactly the width where the spike appears. If the spike is present in the
reuse run and absent in the no-reuse run, and the norm curve steps at the same
place, weight norm is implicated.

Two stacked panels sharing a log x-axis of parameter count:
  top     test (marker 'D') and train zero-one loss, both sweeps
  bottom  weight norm, both sweeps -- --norm_metric picks which one

--norm_metric weight_norm (default) is the trained model's norm;
init_weight_norm is the norm before that H's first SGD step, which is the one
that isolates what reuse itself carries in as opposed to what training grows;
hidden_weight_norm / output_weight_norm split the trained norm by layer.

Same argument names and colors as compare_full_sweep_belkin_style.py, which
does the analogous two-summary comparison in Belkin's own Fig. 3 layout.
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from config import K, N_TRAIN
from full_sweep import RESULTS_ROOT
from utils import num_params

OUT_PATH = os.path.join(RESULTS_ROOT, "full_sweep_weight_norm_comparison.png")

INTERPOLATION_THRESHOLD = K * N_TRAIN / 1e3  # params where num_params(H) == K*N_TRAIN

# Okabe-Ito colorblind-safe pair, same assignment as
# compare_full_sweep_belkin_style.py so the two scripts' figures read together.
SWEEP1_COLOR = "#CC79A7"  # reddish purple
SWEEP2_COLOR = "#009E73"  # bluish green

NORM_YLABEL = {
    "weight_norm": "L2 norm of trained weights",
    "init_weight_norm": "L2 norm of weights at initialization",
    "hidden_weight_norm": "L2 norm of trained hidden layer weights",
    "output_weight_norm": "L2 norm of trained output layer weights",
}


def h_vals_in(summary, column):
    # Anchored, because the norm column names are suffixes of each other --
    # a plain endswith("_weight_norm_mean") also matches init_weight_norm,
    # hidden_weight_norm and output_weight_norm.
    col_re = re.compile(rf"^H(\d+)_{re.escape(column)}_mean$")
    return sorted(int(m.group(1)) for c in summary.columns if (m := col_re.match(c)))


def load_row(summary_path, lr, batch_size, norm_metric):
    if not os.path.exists(summary_path):
        raise SystemExit(f"{summary_path} not found -- run aggregate_full_sweep.py first.")
    summary = pd.read_csv(summary_path)
    row_match = summary[(summary["lr"] == lr) & (summary["batch_size"] == batch_size)]
    if row_match.empty:
        raise SystemExit(f"No row with lr={lr}, batch_size={batch_size} in {summary_path}.")

    h_vals = h_vals_in(summary, "test_zeroone")
    norm_h_vals = h_vals_in(summary, norm_metric)
    if not norm_h_vals:
        raise SystemExit(f"No H*_{norm_metric}_mean columns in {summary_path} -- that sweep "
                          f"predates weight-norm recording, so it has to be re-run.")
    # An H present for the error curves but not for the norm (or vice versa)
    # would silently misalign the two panels, so plot only what both have.
    return row_match.iloc[0], [h for h in h_vals if h in set(norm_h_vals)]


def plot_sweep(ax_top, ax_bot, row, h_vals, norm_metric, color, label, linestyle):
    # The second sweep is dashed, not just recolored: a belkin/belkin_noreuse
    # pair differs ONLY on the underparameterized side (Belkin's overparam
    # reuse flag is already False), so the two curves coincide exactly above
    # the threshold and a solid line drawn second would completely hide the
    # first one there.
    x = [num_params(h) / 1e3 for h in h_vals]
    ax_top.plot(x, [row[f"H{h}_test_zeroone_mean"] * 100 for h in h_vals],
                marker="D", ms=4, color=color, linestyle=linestyle, label=f"{label} test")
    ax_top.plot(x, [row[f"H{h}_train_zeroone_mean"] * 100 for h in h_vals],
                color=color, linestyle=linestyle, label=f"{label} train")
    ax_bot.plot(x, [row[f"H{h}_{norm_metric}_mean"] for h in h_vals],
                marker="D", ms=4, color=color, linestyle=linestyle, label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary_path1", required=True, help="First summary CSV to plot.")
    parser.add_argument("--summary_path2", required=True, help="Second summary CSV to plot.")
    parser.add_argument("--sweep_name1", required=True, help="Legend label for the first summary.")
    parser.add_argument("--sweep_name2", required=True, help="Legend label for the second summary.")
    parser.add_argument("--lr1", type=float, default=0.0005, help="lr row to pull from summary_path1 (default: 0.0005).")
    parser.add_argument("--batch_size1", type=int, default=32, help="batch_size row to pull from summary_path1 (default: 32).")
    parser.add_argument("--lr2", type=float, default=0.0005, help="lr row to pull from summary_path2 (default: 0.0005).")
    parser.add_argument("--batch_size2", type=int, default=32, help="batch_size row to pull from summary_path2 (default: 32).")
    parser.add_argument("--norm_metric", choices=list(NORM_YLABEL), default="weight_norm",
                         help="Which recorded norm to plot on the bottom panel (default: weight_norm, "
                              "the trained model's).")
    parser.add_argument("--log_norm", action="store_true",
                         help="Log-scale the weight-norm axis -- use this if one sweep's norms are "
                              "orders of magnitude larger and flatten the other's curve.")
    parser.add_argument("--out_path", default=OUT_PATH, help="Path to save the plot to.")
    parser.add_argument("--plot_title", default=None,
                         help="Plot title (default: '{sweep_name1} vs {sweep_name2} weight norm').")
    args = parser.parse_args()

    row1, h_vals1 = load_row(args.summary_path1, args.lr1, args.batch_size1, args.norm_metric)
    row2, h_vals2 = load_row(args.summary_path2, args.lr2, args.batch_size2, args.norm_metric)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    plot_sweep(ax_top, ax_bot, row1, h_vals1, args.norm_metric, SWEEP1_COLOR, args.sweep_name1, "-")
    plot_sweep(ax_top, ax_bot, row2, h_vals2, args.norm_metric, SWEEP2_COLOR, args.sweep_name2, "--")

    ax_top.axvline(INTERPOLATION_THRESHOLD, color="black", linestyle=":", alpha=0.5)
    ax_top.set_ylim(bottom=0)
    ax_top.set_ylabel("Zero-one loss (%)")
    ax_top.legend(fontsize=7, ncol=2)

    ax_bot.axvline(INTERPOLATION_THRESHOLD, color="black", linestyle=":", alpha=0.5)
    if args.log_norm:
        ax_bot.set_yscale("log")
    else:
        ax_bot.set_ylim(bottom=0)
    ax_bot.set_ylabel(NORM_YLABEL[args.norm_metric])
    ax_bot.set_xlabel(r"Number of parameters/weights ($\times10^3$)")
    ax_bot.legend(fontsize=7)

    if args.plot_title is None:
        ax_top.set_title(f"{args.sweep_name1} vs {args.sweep_name2} weight norm")
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
