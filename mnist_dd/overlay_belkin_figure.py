"""Overlay our own results on top of belkin_figure3.png, axes aligned, both panels.

Uses the pixel<->data calibration from calibrate_belkin_figure.py (run that
first) to crop the image to each panel and place it at the correct extent on
a log-x/linear-y axes, then plots our own mean test curves on top for a
direct visual comparison -- since we don't have Belkin's raw numbers, this is
a qualitative overlay, not a numeric fit.

Expects a summary CSV with one row per (lr, batch_size) and columns
H{h}_test_zeroone_mean and H{h}_test_MSE_mean for each H in H_VALS, i.e. the
shape produced by aggregate_full_sweep.py.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from config import H_VALS
from utils import num_params

CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "belkin_calibration.json")
FIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "belkin_figure3.png")
DEFAULT_SUMMARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "full_sweep_summary.csv")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "overlay_vs_belkin.png")


def panel_extent(crop_box, y_fit, x_fit):
    left, top, right, bottom = crop_box

    def px_to_x(px):
        return 10 ** (x_fit[0] * px + x_fit[1]) * 1000  # raw param count

    def py_to_y(py):
        return y_fit[0] * py + y_fit[1]

    x_left, x_right = px_to_x(left), px_to_x(right)
    y_top, y_bottom = py_to_y(top), py_to_y(bottom)
    return x_left, x_right, y_bottom, y_top


def draw_panel(ax, crop, extent, x, summary, value_col_tmpl, colors, ylabel, scale=1.0):
    x_left, x_right, y_bottom, y_top = extent
    ax.imshow(crop, extent=[x_left, x_right, y_bottom, y_top], aspect="auto", zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    for i, (_, row) in enumerate(summary.iterrows()):
        y = [row[value_col_tmpl.format(h=h)] * scale for h in H_VALS]
        ax.plot(
            x, y, marker="o", ms=4, linewidth=1.5, zorder=5,
            color=colors[i % len(colors)],
            label=f"ours: lr={row['lr']}, bs={row['batch_size']}",
        )
    ax.set_ylabel(ylabel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_path", default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args()

    if not os.path.exists(CALIBRATION_PATH):
        raise SystemExit(f"{CALIBRATION_PATH} not found -- run calibrate_belkin_figure.py first.")
    if not os.path.exists(args.summary_path):
        raise SystemExit(f"{args.summary_path} not found -- run aggregate_full_sweep.py first.")

    with open(CALIBRATION_PATH) as f:
        cal = json.load(f)
    x_fit = cal["x_fit"]

    img = Image.open(FIG_PATH).convert("RGB")

    def crop_for(box):
        l, t, r, b = [int(round(v)) for v in box]
        return np.array(img.crop((l, t, r, b)))

    crop_top = crop_for(cal["crop_box_top"])
    crop_bottom = crop_for(cal["crop_box_bottom"])
    extent_top = panel_extent(cal["crop_box_top"], cal["y_fit_top"], x_fit)
    extent_bottom = panel_extent(cal["crop_box_bottom"], cal["y_fit_bottom"], x_fit)

    summary = pd.read_csv(args.summary_path)
    x = [num_params(h) for h in H_VALS]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 10))
    draw_panel(ax_top, crop_top, extent_top, x, summary, "H{h}_test_zeroone_mean", colors,
               "Zero-one loss (%)", scale=100.0)
    ax_top.legend(fontsize=8)
    ax_top.set_title("Our results overlaid on Belkin Fig. 3 (background = his, markers = ours)")

    draw_panel(ax_bottom, crop_bottom, extent_bottom, x, summary, "H{h}_test_MSE_mean", colors,
               "Squared loss")
    ax_bottom.set_xlabel("Number of parameters/weights")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
