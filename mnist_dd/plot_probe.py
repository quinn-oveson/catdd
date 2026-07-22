"""Plot Stage 1 probe results, for visual comparison against Belkin's small-H targets.

Reads results/probe_summary.csv (built by aggregate_probe.py -- run that first). Plots each
(lr, batch_size) candidate's mean test zero-one loss at PROBE_H_VALS (x-axis log scale, to
emulate Belkin's own plot), plus Belkin's approximate small-H target values as a reference --
we don't have Belkin's raw underlying numbers, only the published figure, so this is a
qualitative comparison, not a numeric fit. (Belkin's actual figure isn't overlaid here since its
H/parameter-count axis doesn't match PROBE_H_VALS -- see belkin_figure3.png directly instead.)
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from probe import PROBE_H_VALS, PROBE_H_TARGET_ZEROONE_PCT, RESULTS_DIR
from utils import num_params

SUMMARY_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "probe_summary.csv")
OUT_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "probe_vs_belkin.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_path", default=SUMMARY_PATH,
                         help="Which probe summary CSV to plot (default: results/probe_summary.csv).")
    args = parser.parse_args()

    if not os.path.exists(args.summary_path):
        raise SystemExit(f"{args.summary_path} not found -- run aggregate_probe.py first.")
    summary = pd.read_csv(args.summary_path)

    fig, ax_probe = plt.subplots(figsize=(10, 5))

    x = [num_params(h) for h in PROBE_H_VALS]
    for _, row in summary.iterrows():
        y = [row[f"H{h}_test_zeroone_mean"] * 100 for h in PROBE_H_VALS]
        yerr = [row[f"H{h}_test_zeroone_std"] * 100 for h in PROBE_H_VALS]
        ax_probe.errorbar(x, y, yerr=yerr, marker="D", ms=4, capsize=3,
                           label=f"lr={row['lr']}, bs={row['batch_size']}")

    ax_probe.plot(x, PROBE_H_TARGET_ZEROONE_PCT, "k--", marker="*", ms=12,
                  label="Belkin target (approx.)")

    ax_probe.set_xscale("log")
    ax_probe.set_xlim(right=800000)
    ax_probe.set_ylim(bottom=0)
    ax_probe.set_xlabel("Number of parameters (log scale)")
    ax_probe.set_ylabel("Zero-one loss (%)")
    ax_probe.set_title("Stage 1 probe: mean test zero-one loss (±1 std across seeds)")
    ax_probe.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    if args.summary_path == SUMMARY_PATH:
        out_path = OUT_PATH
    else:
        stem = os.path.splitext(os.path.basename(args.summary_path))[0]
        out_path = os.path.join(os.path.dirname(RESULTS_DIR), f"probe_vs_belkin_{stem}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
