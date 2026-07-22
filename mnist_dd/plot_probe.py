"""Plot Stage 1 probe results, for visual comparison against Belkin's full curve.

Reads results/probe_summary.csv (built by aggregate_probe.py -- run that first) and
plots each (lr, batch_size) candidate's mean test zero-one loss at PROBE_H_VALS,
against Belkin's full test zero-one loss curve across the entire H_VALS range --
digitized directly off his Fig. 3 by clicking every marker (see
extract_belkin_markers.py / results/belkin_digitized.csv), not eyeballed -- so the
small-H probe candidates can be judged against the complete double-descent shape,
not just a few isolated target points.
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from config import K, N_TRAIN
from probe import PROBE_H_VALS, RESULTS_DIR
from utils import num_params

INTERPOLATION_THRESHOLD = K * N_TRAIN / 1e3  # params where num_params(H) == K*N_TRAIN

SUMMARY_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "probe_summary.csv")
BELKIN_DIGITIZED_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "belkin_digitized.csv")
OUT_PATH = os.path.join(os.path.dirname(RESULTS_DIR), "probe_vs_belkin.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_path", default=SUMMARY_PATH,
                         help="Which probe summary CSV to plot (default: results/probe_summary.csv).")
    parser.add_argument("--no_errorbars", action="store_true",
                         help="Skip std error bars -- use this if the summary CSV doesn't have "
                              "std for every cell (e.g. some cells only had 1 seed).")
    args = parser.parse_args()

    if not os.path.exists(args.summary_path):
        raise SystemExit(f"{args.summary_path} not found -- run aggregate_probe.py first.")
    if not os.path.exists(BELKIN_DIGITIZED_PATH):
        raise SystemExit(f"{BELKIN_DIGITIZED_PATH} not found -- run extract_belkin_markers.py first.")
    summary = pd.read_csv(args.summary_path)
    belkin = pd.read_csv(BELKIN_DIGITIZED_PATH).sort_values("H")

    fig, ax_probe = plt.subplots(figsize=(10, 5))

    x_probe = [num_params(h) / 1e3 for h in PROBE_H_VALS]
    for _, row in summary.iterrows():
        y = [row[f"H{h}_test_zeroone_mean"] * 100 for h in PROBE_H_VALS]
        if args.no_errorbars:
            ax_probe.plot(x_probe, y, marker="D", ms=4,
                          label=f"lr={row['lr']}, bs={row['batch_size']}")
        else:
            yerr = [row[f"H{h}_test_zeroone_std"] * 100 for h in PROBE_H_VALS]
            ax_probe.errorbar(x_probe, y, yerr=yerr, marker="D", ms=4, capsize=3,
                               label=f"lr={row['lr']}, bs={row['batch_size']}")

    x_belkin = belkin["N"] / 1e3
    ax_probe.plot(x_belkin, belkin["test_zeroone_pct"], "k--", marker="*", ms=8,
                  label="Belkin (digitized from Fig. 3)")

    ax_probe.axvline(INTERPOLATION_THRESHOLD, color="gray", linestyle=":", linewidth=1.5,
                      label="Interpolation threshold")

    ax_probe.set_xscale("log")
    ax_probe.set_ylim(bottom=0)
    ax_probe.set_xlabel(r"Number of parameters/weights ($\times10^3$)")
    ax_probe.set_ylabel("Zero-one loss (%)")
    stage_label = "Stage 2 probe" if args.summary_path == SUMMARY_PATH else "Stage 1 probe"
    title_suffix = "" if args.no_errorbars else " (±1 std across seeds)"
    ax_probe.set_title(f"{stage_label} vs. Belkin's full digitized curve{title_suffix}")
    ax_probe.legend(fontsize=7, ncol=2)

    ticks = [3, 10, 40, 100, 300, 800]
    ax_probe.set_xticks(ticks)
    ax_probe.xaxis.set_major_formatter(ScalarFormatter())
    ax_probe.ticklabel_format(style="plain", axis="x")

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
