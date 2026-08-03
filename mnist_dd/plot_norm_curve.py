"""Plot one weight-norm metric against parameter count for several full-sweep
summaries, on a single panel.

Unlike plot_weight_norm_comparison.py (error on top, norm on bottom, two
sweeps), this draws just the norm curve, so init and trained norms can be
separate figures. Colors come from compare_full_sweep_belkin_style.SWEEP_COLORS
in the order the summaries are given, so a sweep keeps its color across all of
these figures.

    python plot_norm_curve.py --norm_metric init_weight_norm \
        --summary_path results/full_sweep_summary_belkin.csv --sweep_name "Weight reuse" \
        --summary_path results/full_sweep_summary_belkin_noreuse.csv --sweep_name "No reuse"
"""
import argparse
import os
import re

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from compare_full_sweep_belkin_style import SWEEP_COLORS, SWEEP_LINESTYLES
from config import K, N_TRAIN
from full_sweep import RESULTS_ROOT
from utils import num_params

INTERPOLATION_THRESHOLD = K * N_TRAIN / 1e3

YLABEL = {
    "init_weight_norm": "L2 norm of weights at initialization",
    "weight_norm": "L2 norm of trained weights",
    "hidden_weight_norm": "L2 norm of trained hidden layer weights",
    "output_weight_norm": "L2 norm of trained output layer weights",
}


def h_vals_in(summary, column):
    # Anchored: the norm column names are suffixes of each other, so a plain
    # endswith("_weight_norm_mean") also matches init/hidden/output.
    col_re = re.compile(rf"^H(\d+)_{re.escape(column)}_mean$")
    return sorted(int(m.group(1)) for c in summary.columns if (m := col_re.match(c)))


def load_row(summary_path, lr, batch_size, metric):
    if not os.path.exists(summary_path):
        raise SystemExit(f"{summary_path} not found -- run aggregate_full_sweep.py first.")
    summary = pd.read_csv(summary_path)
    match = summary[(summary["lr"] == lr) & (summary["batch_size"] == batch_size)]
    if match.empty:
        raise SystemExit(f"No row with lr={lr}, batch_size={batch_size} in {summary_path}.")
    h_vals = h_vals_in(summary, metric)
    if not h_vals:
        raise SystemExit(f"No H*_{metric}_mean columns in {summary_path} -- that sweep "
                          f"predates weight-norm recording, so it has to be re-run.")
    return match.iloc[0], h_vals



def taper(i, n, hi, lo):
    """Value for series i of n, decreasing hi -> lo in draw order. These arms
    coincide exactly over parts of the H range, so later curves are drawn
    thinner and sit visibly on top of earlier ones."""
    return hi if n < 2 else hi + (lo - hi) * i / (n - 1)


def parse_linestyle(s):
    """CLI linestyle: matplotlib's own ("-", "--", ":") or a dash pattern
    ("6,1.5" -> (0, (6, 1.5))), which the named styles can't express."""
    if "," in s:
        return (0, tuple(float(x) for x in s.split(",")))
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary_path", action="append", required=True, metavar="PATH",
                         help="Summary CSV to plot. Repeat once per sweep.")
    parser.add_argument("--sweep_name", action="append", required=True, metavar="NAME",
                         help="Legend label for the corresponding --summary_path.")
    parser.add_argument("--color", action="append", metavar="HEX",
                         help="Color per sweep, in order. Defaults to SWEEP_COLORS.")
    parser.add_argument("--linestyle", action="append", metavar="STYLE",
                         help="Linestyle per sweep, in order (see parse_linestyle). "
                              "Defaults to SWEEP_LINESTYLES.")
    parser.add_argument("--norm_metric", choices=list(YLABEL), default="init_weight_norm",
                         help="Which recorded norm to plot (default: init_weight_norm).")
    parser.add_argument("--lr", type=float, default=0.0005, help="lr row to pull (default: 0.0005).")
    parser.add_argument("--batch_size", type=int, default=32,
                         help="batch_size row to pull (default: 32; use 4000 for --full_batch runs).")
    parser.add_argument("--stroke", action="store_true",
                         help="Outline lines/markers in dark gray, so bright colors "
                              "(the warm end of plasma/viridis) stay legible on white.")
    parser.add_argument("--std_band", action="store_true",
                         help="Shade +/-1 std across seeds around each curve.")
    parser.add_argument("--log_norm", action="store_true", help="Log-scale the y axis.")
    parser.add_argument("--out_path", default=None,
                         help="Default: results/full_sweep_<norm_metric>_curve.png")
    parser.add_argument("--plot_title", default=None)
    args = parser.parse_args()

    if len(args.summary_path) != len(args.sweep_name):
        raise SystemExit(f"got {len(args.summary_path)} --summary_path but "
                          f"{len(args.sweep_name)} --sweep_name -- pass one name per path.")
    colors = args.color or SWEEP_COLORS
    styles = [parse_linestyle(s) for s in (args.linestyle or ["-"] * len(colors))]
    if len(args.summary_path) > min(len(colors), len(styles)):
        raise SystemExit(f"{len(args.summary_path)} summaries but only {len(colors)} colors / "
                          f"{len(styles)} linestyles -- pass --color/--linestyle for each.")

    fig, ax = plt.subplots(figsize=(10, 5.5))

    n = len(args.summary_path)
    for i, (path, name, color, ls) in enumerate(zip(args.summary_path, args.sweep_name,
                                                    colors, styles)):
        row, h_vals = load_row(path, args.lr, args.batch_size, args.norm_metric)
        x = [num_params(h) / 1e3 for h in h_vals]
        mean = [row[f"H{h}_{args.norm_metric}_mean"] for h in h_vals]
        lw = taper(i, n, 2.8, 1.4)
        fx = [pe.Stroke(linewidth=lw + 1.6, foreground="#3a3a3a"), pe.Normal()] if args.stroke else None
        ax.plot(x, mean, marker="D", ms=taper(i, n, 6.5, 3.5), lw=lw,
                color=color, linestyle=ls, label=name, path_effects=fx,
                markeredgecolor="#3a3a3a" if args.stroke else color,
                markeredgewidth=0.7 if args.stroke else 0)
        if args.std_band:
            std = [row.get(f"H{h}_{args.norm_metric}_std", 0) for h in h_vals]
            ax.fill_between(x, [m - s for m, s in zip(mean, std)],
                             [m + s for m, s in zip(mean, std)], color=color, alpha=0.15, lw=0)

    ax.axvline(INTERPOLATION_THRESHOLD, color="black", linestyle=":", alpha=0.5)
    if args.log_norm:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_ylabel(YLABEL[args.norm_metric])
    ax.set_xlabel(r"Number of parameters/weights ($\times10^3$)")
    ax.set_title(args.plot_title if args.plot_title else YLABEL[args.norm_metric])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.set_xscale("log")
    ax.set_xticks([3, 10, 40, 100, 300, 800])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="x")

    out_path = args.out_path or os.path.join(
        RESULTS_ROOT, f"full_sweep_{args.norm_metric}_curve.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
