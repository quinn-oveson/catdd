"""Interactive pixel<->data calibration for belkin_figure3.png (both panels).

Belkin's Fig. 3 gives us a picture, not numbers: no table of N or H values for
the fully-connected MNIST experiment appears anywhere in the paper or SI
appendix (checked both in full) -- only the figure itself, with a log-scaled
x-axis (parameters/weights, x10^3) shared by both panels, and two different
linear y-axes: zero-one loss % (top panel) and squared loss (bottom panel).
This script lets you click on known tick marks so we can fit a pixel->data
mapping for both panels at once, then reuses that mapping in two other
scripts:
  - extract_belkin_markers.py: read off Belkin's own marker positions (his
    approximate H values) from either/both panels, by detecting his blue
    diamond markers in pixel space and converting to data space.
  - overlay_belkin_figure.py: draw our own results on top of his figure with
    correctly aligned axes, for both zero-one loss and squared loss.

Run this LOCALLY (needs a GUI backend / real display, e.g. on your laptop --
not over SSH on the cluster). It opens the image and asks for 18 clicks, in
this exact order:
  1-6.   The six labeled x-axis tick marks (shared by both panels, only need
         to click them once), left to right: 3, 10, 40, 100, 300, 800
         (click on the tick mark itself, near the bottom axis).
  7-10.  The four labeled y-axis tick marks of the TOP panel (zero-one loss %),
         top to bottom: 60, 40, 20, 0.
  11-14. The four labeled y-axis tick marks of the BOTTOM panel (squared loss),
         top to bottom: 0.6, 0.4, 0.2, 0.0.
  15.    Top-left corner of the TOP panel's plot box.
  16.    Bottom-right corner of the TOP panel's plot box.
  17.    Top-left corner of the BOTTOM panel's plot box.
  18.    Bottom-right corner of the BOTTOM panel's plot box.

Calibration is saved to results/belkin_calibration.json and only needs to be
done once (redo it if you crop/replace belkin_figure3.png).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

FIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "belkin_figure3.png")
CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "belkin_calibration.json")

X_TICK_VALUES = [3, 10, 40, 100, 300, 800]  # x1000 params, shared axis
Y_TICK_VALUES_TOP = [60, 40, 20, 0]  # zero-one loss %, top-to-bottom
Y_TICK_VALUES_BOTTOM = [0.6, 0.4, 0.2, 0.0]  # squared loss, top-to-bottom


def fit_linear(pixels, values):
    return list(np.polyfit(np.array(pixels), np.array(values, dtype=float), 1))


def main():
    img = np.array(Image.open(FIG_PATH).convert("RGB"))

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.imshow(img)
    ax.set_title(
        "Click in order: x-ticks 3,10,40,100,300,800 -> top y-ticks 60,40,20,0 -> "
        "bottom y-ticks .6,.4,.2,0 -> top box TL,BR -> bottom box TL,BR",
        fontsize=7,
    )
    fig.tight_layout()

    print(__doc__)
    print("Click 18 points on the figure window now...")
    pts = plt.ginput(18, timeout=0)
    plt.close(fig)

    x_clicks = pts[0:6]
    y_clicks_top = pts[6:10]
    y_clicks_bottom = pts[10:14]
    top_tl, top_br = pts[14], pts[15]
    bottom_tl, bottom_br = pts[16], pts[17]

    px = [p[0] for p in x_clicks]
    log_vals = np.log10(X_TICK_VALUES).tolist()
    x_fit = fit_linear(px, log_vals)  # log10(value) = x_fit[0]*px + x_fit[1]

    py_top = [p[1] for p in y_clicks_top]
    y_fit_top = fit_linear(py_top, Y_TICK_VALUES_TOP)  # value = y_fit_top[0]*py + y_fit_top[1]

    py_bottom = [p[1] for p in y_clicks_bottom]
    y_fit_bottom = fit_linear(py_bottom, Y_TICK_VALUES_BOTTOM)

    print("\nCalibration check (predicted vs. target value for each click):")
    print("  x-ticks (shared):")
    for p, v in zip(px, X_TICK_VALUES):
        pred = 10 ** (x_fit[0] * p + x_fit[1])
        print(f"    clicked px={p:7.1f}  target={v:6.1f}  fit predicts={pred:7.2f}")
    print("  top panel y-ticks (zero-one loss %):")
    for p, v in zip(py_top, Y_TICK_VALUES_TOP):
        pred = y_fit_top[0] * p + y_fit_top[1]
        print(f"    clicked px={p:7.1f}  target={v:6.2f}  fit predicts={pred:7.2f}")
    print("  bottom panel y-ticks (squared loss):")
    for p, v in zip(py_bottom, Y_TICK_VALUES_BOTTOM):
        pred = y_fit_bottom[0] * p + y_fit_bottom[1]
        print(f"    clicked px={p:7.1f}  target={v:6.2f}  fit predicts={pred:7.2f}")
    print(
        "\nIf any 'fit predicts' value is far off its target, your clicks were "
        "imprecise -- rerun the script and click more carefully on the tick marks."
    )

    calibration = {
        "x_fit": x_fit,
        "y_fit_top": y_fit_top,
        "y_fit_bottom": y_fit_bottom,
        "crop_box_top": [top_tl[0], top_tl[1], top_br[0], top_br[1]],
        "crop_box_bottom": [bottom_tl[0], bottom_tl[1], bottom_br[0], bottom_br[1]],
    }

    os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\nWrote calibration to {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
