"""Plot Stage 2 full-sweep results, for visual comparison against Belkin's full curve.

Reads results/full_sweep_summary.csv (built by aggregate_full_sweep.py -- run that
first) and plots each (lr, batch_size) candidate's mean test zero-one loss at
whichever H's are present in the summary, against Belkin's full test zero-one loss
curve digitized off his Fig. 3 (results/belkin_digitized.csv). Same format as
plot_probe.py, but a separate script/output so it doesn't overwrite
results/probe_vs_belkin.png -- the two summaries have different H columns (probe
only covers PROBE_H_VALS; a full-sweep summary may only have the independent
overparameterized H's aggregated so far, before the chain/underparameterized
H's are done), so H's to plot are read directly off the summary's own columns
rather than assumed from config.H_VALS.
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
OUT_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "full_sweep_vs_belkin.png")

INTERPOLATION_THRESHOLD = K * N_TRAIN / 1e3  # params where num_params(H) == K*N_TRAIN

H_COLUMN_RE = re.compile(r"^H(\d+)_test_zeroone_mean$")

# Matches the colors lr=0.001/0.0005 happen to land on in probe_vs_belkin.png
# (matplotlib's default tab10 cycle, keyed explicitly here since full_sweep's
# summary only has these two lr rows -- relying on cycle order wouldn't match).
LR_COLORS = {0.001: "tab:green", 0.0005: "tab:orange"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_path", default=SUMMARY_PATH,
                         help="Which full-sweep summary CSV to plot (default: results/full_sweep_summary.csv).")
    parser.add_argument("--plot_title", default=None,
                         help="What title to give this plot (default: Stage 3 full sweep vs. Belkin's full digitized curve).")
    parser.add_argument("--no_errorbars", action="store_true",
                         help="Skip std error bars -- use this if the summary CSV doesn't have "
                              "std for every cell (e.g. some cells only had 1 seed).")
    args = parser.parse_args()

    if not os.path.exists(args.summary_path):
        raise SystemExit(f"{args.summary_path} not found -- run aggregate_full_sweep.py first.")
    if not os.path.exists(BELKIN_DIGITIZED_PATH):
        raise SystemExit(f"{BELKIN_DIGITIZED_PATH} not found -- run extract_belkin_markers.py first.")
    summary = pd.read_csv(args.summary_path)
    belkin = pd.read_csv(BELKIN_DIGITIZED_PATH).sort_values("H")

    h_vals = sorted(int(m.group(1)) for c in summary.columns if (m := H_COLUMN_RE.match(c)))
    if not h_vals:
        raise SystemExit(f"No H*_test_zeroone_mean columns found in {args.summary_path}.")

    fig, ax = plt.subplots(figsize=(10, 5))

    x_sweep = [num_params(h) / 1e3 for h in h_vals]
    for _, row in summary.iterrows():
        y = [row[f"H{h}_test_zeroone_mean"] * 100 for h in h_vals]
        color = LR_COLORS.get(row["lr"])
        if args.no_errorbars:
            ax.plot(x_sweep, y, marker="D", ms=4, color=color,
                    label=f"lr={row['lr']}, bs={row['batch_size']}")
        else:
            yerr = [row[f"H{h}_test_zeroone_std"] * 100 for h in h_vals]
            ax.errorbar(x_sweep, y, yerr=yerr, marker="D", ms=4, capsize=3, color=color,
                        label=f"lr={row['lr']}, bs={row['batch_size']}")

    x_belkin = belkin["N"] / 1e3
    ax.plot(x_belkin, belkin["test_zeroone_pct"], "k--", marker="*", ms=8,
            label="Belkin (digitized from Fig. 3)")

    ax.axvline(INTERPOLATION_THRESHOLD, color="gray", linestyle=":", linewidth=1.5,
               label="Interpolation threshold")

    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Number of parameters/weights ($\times10^3$)")
    ax.set_ylabel("Zero-one loss (%)")
    title_suffix = "" if args.no_errorbars else " (±1 std across seeds)"
    if args.plot_title == None:
        ax.set_title(f"Stage 3 full sweep vs. Belkin's full digitized curve {title_suffix}")
    else:
        ax.set_title(args.plot_title)
    ax.legend(fontsize=7, ncol=2)

    ticks = [3, 10, 40, 100, 300, 800]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="x")

    fig.tight_layout()
    if args.summary_path == SUMMARY_PATH:
        out_path = OUT_PATH
    else:
        stem = os.path.splitext(os.path.basename(args.summary_path))[0]
        out_path = os.path.join(os.path.dirname(RESULTS_DIR), f"full_sweep_vs_belkin_{stem}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
