"""Plot Stage 2 full-sweep results, for visual comparison against Belkin's full curve
(or against each other, with --include_belkin=False).

Reads results/full_sweep_summary.csv (built by aggregate_full_sweep.py -- run that
first) and plots each (lr, batch_size) candidate's mean test loss at whichever H's
are present in the summary, against Belkin's full test zero-one loss curve digitized
off his Fig. 3 (results/belkin_digitized.csv). Same format as plot_probe.py, but a
separate script/output so it doesn't overwrite results/probe_vs_belkin.png -- the two
summaries have different H columns (probe only covers PROBE_H_VALS; a full-sweep
summary may only have the independent overparameterized H's aggregated so far,
before the chain/underparameterized H's are done), so H's to plot are read directly
off the summary's own columns rather than assumed from config.H_VALS.

--metric selects which loss to plot (zeroone or CE); Belkin's digitized curve is
only available in zeroone terms, so --include_belkin=True requires --metric=zeroone.
Candidate lines use matplotlib's default color cycle rather than a fixed per-lr
color, since which lr values are present varies by summary CSV.
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from config import K, N_TRAIN
from full_sweep import RESULTS_DIR
from utils import num_params

SUMMARY_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "full_sweep_summary.csv")
BELKIN_DIGITIZED_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "belkin_digitized.csv")

INTERPOLATION_THRESHOLD = K * N_TRAIN / 1e3  # params where num_params(H) == K*N_TRAIN

METRIC_YLABEL = {"zeroone": "Zero-one loss (%)", "CE": "Cross entropy loss"}
METRIC_SCALE = {"zeroone": 100, "CE": 1}  # zeroone stored as a fraction; CE already in its native units


def str2bool(v):
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected a bool (True/False), got {v!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_path", default=SUMMARY_PATH,
                         help="Which full-sweep summary CSV to plot (default: results/full_sweep_summary.csv).")
    parser.add_argument("--metric", choices=["zeroone", "CE"], default="zeroone",
                         help="Which test loss to plot (default: zeroone).")
    parser.add_argument("--include_belkin", type=str2bool, default=None,
                         help="Overlay Belkin's digitized Fig. 3 curve (default: True for --metric=zeroone, "
                              "False otherwise -- Belkin's curve is only digitized in zeroone terms).")
    parser.add_argument("--plot_title", default=None,
                         help="What title to give this plot (default: Stage 3 full sweep, optionally "
                              "vs. Belkin's full digitized curve).")
    parser.add_argument("--no_errorbars", action="store_true",
                         help="Skip std error bars -- use this if the summary CSV doesn't have "
                              "std for every cell (e.g. some cells only had 1 seed).")
    parser.add_argument("--exclude_lr", type=float, nargs="*", default=[],
                         help="lr values to drop from the plot (e.g. an unstable high-lr candidate "
                              "whose error bars dwarf the rest of the grid).")
    parser.add_argument("--only_lr_bs", type=str, nargs="*", default=[],
                         help="Restrict the plot to exactly these (lr, batch_size) candidates, given as "
                              "'lr:batch_size' pairs (e.g. --only_lr_bs 0.0005:800 0.05:32) -- for "
                              "highlighting a few standout combos instead of the whole grid.")
    args = parser.parse_args()

    include_belkin = (args.metric == "zeroone") if args.include_belkin is None else args.include_belkin
    if include_belkin and args.metric != "zeroone":
        raise SystemExit("--include_belkin=True requires --metric=zeroone -- Belkin's digitized curve "
                          "only has zeroone loss.")

    if not os.path.exists(args.summary_path):
        raise SystemExit(f"{args.summary_path} not found -- run aggregate_full_sweep.py first.")
    summary = pd.read_csv(args.summary_path)
    if args.exclude_lr:
        summary = summary[~summary["lr"].isin(args.exclude_lr)]
        if summary.empty:
            raise SystemExit(f"--exclude_lr {args.exclude_lr} dropped every row in {args.summary_path}.")
    if args.only_lr_bs:
        pairs = [tuple(p.split(":")) for p in args.only_lr_bs]
        pairs = [(float(lr), int(bs)) for lr, bs in pairs]
        mask = summary.apply(lambda row: (row["lr"], int(row["batch_size"])) in pairs, axis=1)
        summary = summary[mask]
        if summary.empty:
            raise SystemExit(f"--only_lr_bs {args.only_lr_bs} matched no rows in {args.summary_path}.")
    if include_belkin:
        if not os.path.exists(BELKIN_DIGITIZED_PATH):
            raise SystemExit(f"{BELKIN_DIGITIZED_PATH} not found -- run extract_belkin_markers.py first.")
        belkin = pd.read_csv(BELKIN_DIGITIZED_PATH).sort_values("H")

    h_column_re = re.compile(rf"^H(\d+)_test_{args.metric}_mean$")
    h_vals = sorted(int(m.group(1)) for c in summary.columns if (m := h_column_re.match(c)))
    if not h_vals:
        raise SystemExit(f"No H*_test_{args.metric}_mean columns found in {args.summary_path}.")

    fig, ax = plt.subplots(figsize=(10, 5))

    scale = METRIC_SCALE[args.metric]
    x_sweep = [num_params(h) / 1e3 for h in h_vals]
    for _, row in summary.iterrows():
        y = [row[f"H{h}_test_{args.metric}_mean"] * scale for h in h_vals]
        label = f"lr={row['lr']}, bs={row['batch_size']}"
        if args.no_errorbars:
            ax.plot(x_sweep, y, marker="D", ms=4, label=label)
        else:
            yerr = [row[f"H{h}_test_{args.metric}_std"] * scale for h in h_vals]
            ax.errorbar(x_sweep, y, yerr=yerr, marker="D", ms=4, capsize=3, label=label)

    if include_belkin:
        x_belkin = belkin["N"] / 1e3
        ax.plot(x_belkin, belkin["test_zeroone_pct"], "k--", marker="*", ms=8,
                label="Belkin (digitized from Fig. 3)")

    ax.axvline(INTERPOLATION_THRESHOLD, color="gray", linestyle=":", linewidth=1.5,
               label="Interpolation threshold")

    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Number of parameters/weights ($\times10^3$)")
    ax.set_ylabel(METRIC_YLABEL[args.metric])
    title_suffix = "" if args.no_errorbars else " (±1 std across seeds)"
    if args.plot_title is None:
        title = "Stage 3 full sweep vs. Belkin's full digitized curve" if include_belkin else "Stage 3 full sweep"
        ax.set_title(f"{title}{title_suffix}")
    else:
        ax.set_title(args.plot_title)
    ax.legend(fontsize=7, ncol=2)

    ticks = [3, 10, 40, 100, 300, 800]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="x")

    fig.tight_layout()
    stem = os.path.splitext(os.path.basename(args.summary_path))[0]
    prefix = "full_sweep_vs_belkin" if include_belkin else "full_sweep"
    suffix = "" if args.summary_path == SUMMARY_PATH and args.metric == "zeroone" else f"_{stem}_{args.metric}"
    out_path = os.path.join(os.path.dirname(RESULTS_DIR), f"{prefix}{suffix}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
