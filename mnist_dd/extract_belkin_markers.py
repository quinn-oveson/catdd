"""Digitize Belkin's Fig. 3 Test curves by clicking every marker by hand.

Neither the main paper nor the SI appendix states an explicit N/H list or any
risk values for the fully-connected MNIST experiment (Fig. 3 / Fig. S4)
anywhere in the text -- only the figure itself. So the only way to get his
actual numbers is to read them off the image. An automatic color-detection
pass (an earlier version of this script) is unreliable here: in the crowded
middle cluster (H~25-51, squeezed into a small slice of the log-x-axis) the
diamond markers are only a few pixels apart -- close to or smaller than the
marker glyphs themselves -- so peak-picking on pixel columns can merge two
adjacent markers or drift off-center. A human eye (especially zoomed in) does
this far more reliably.

Workflow (needs calibrate_belkin_figure.py run first, and a real display --
not over SSH):
  1. Click all 23 markers on the BOTTOM panel (squared loss) left to right.
     Chosen first because the squared-loss curve has more vertical spread in
     the crowded region, making individual markers easier to tell apart.
  2. From those x pixel positions (the x-axis is shared by both panels), we
     draw thin red guide lines on the TOP panel and save it as
     results/belkin_top_panel_guides.png -- then show you that guided image.
  3. Click all 23 markers on the TOP panel (zero-one loss), using the guide
     lines to know exactly which crowded diamond you're looking for.

Between clicks you can pan/zoom with the matplotlib toolbar -- just remember
to click the toolbar's zoom/pan button OFF again before clicking a marker,
or your click will pan/zoom instead of registering as a point.

Bottom-panel clicks are cached to results/belkin_bottom_clicks.json so a
misclick on the top panel doesn't force you to redo both phases.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from utils import num_params

CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "belkin_calibration.json")
FIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "belkin_figure3.png")
GUIDE_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "belkin_top_panel_guides.png")
BOTTOM_CLICKS_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "belkin_bottom_clicks.json")
DIGITIZED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "belkin_digitized.csv")
BOTTOM_REVIEW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "belkin_bottom_panel_review.png")
TOP_REVIEW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "belkin_top_panel_review.png")

N_MARKERS = 23
REVIEW_CROSS_SIZE = 3  # pixels, half-length of each arm of the review cross


def draw_crosses(base_img, points, size=REVIEW_CROSS_SIZE, color=(255, 0, 0)):
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    for px, py in points:
        draw.line([(px - size, py), (px + size, py)], fill=color, width=1)
        draw.line([(px, py - size), (px, py + size)], fill=color, width=1)
    return img


def crop_with_margin(img, box, margin=15):
    left, top, right, bottom = box
    w, h = img.size
    return img.crop((
        max(0, left - margin), max(0, top - margin),
        min(w, right + margin), min(h, bottom + margin),
    ))


def click_markers(img, crop_box, title, n=N_MARKERS):
    left, top, right, bottom = crop_box
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img)
    ax.set_xlim(left - 15, right + 15)
    ax.set_ylim(bottom + 15, top - 15)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()

    points = []
    print(f"\n{title}")
    for i in range(n):
        print(f"  Click marker {i + 1}/{n}  (zoom/pan first if needed, remember to toggle zoom mode off before clicking)...")
        pt = plt.ginput(1, timeout=0)
        while not pt:
            print("  No click registered -- try again.")
            pt = plt.ginput(1, timeout=0)
        points.append(pt[0])
    plt.close(fig)
    return points


def main():
    if not os.path.exists(CALIBRATION_PATH):
        raise SystemExit(f"{CALIBRATION_PATH} not found -- run calibrate_belkin_figure.py first.")
    with open(CALIBRATION_PATH) as f:
        cal = json.load(f)
    x_fit = cal["x_fit"]
    y_fit_top = cal["y_fit_top"]
    y_fit_bottom = cal["y_fit_bottom"]
    crop_box_top = cal["crop_box_top"]
    crop_box_bottom = cal["crop_box_bottom"]

    img = np.array(Image.open(FIG_PATH).convert("RGB"))

    bottom_clicks = None
    if os.path.exists(BOTTOM_CLICKS_CACHE):
        reuse = input(
            f"Found cached bottom-panel clicks at {BOTTOM_CLICKS_CACHE}. Reuse them "
            "and only redo the top panel? [Y/n]: "
        ).strip().lower()
        if reuse != "n":
            with open(BOTTOM_CLICKS_CACHE) as f:
                bottom_clicks = json.load(f)

    if bottom_clicks is None:
        bottom_clicks = click_markers(
            img, crop_box_bottom,
            f"BOTTOM panel (squared loss) -- click all {N_MARKERS} markers, left to right",
        )
        with open(BOTTOM_CLICKS_CACHE, "w") as f:
            json.dump(bottom_clicks, f)

    bottom_px = np.array([p[0] for p in bottom_clicks])
    bottom_py = np.array([p[1] for p in bottom_clicks])
    order = np.argsort(bottom_px)
    bottom_px, bottom_py = bottom_px[order], bottom_py[order]

    N_vals = 10 ** (x_fit[0] * bottom_px + x_fit[1]) * 1000
    H_vals = (N_vals - 10) / 795
    squared_loss = y_fit_bottom[0] * bottom_py + y_fit_bottom[1]

    print("\nBottom panel digitized (sorted left to right):")
    for h, n, s in zip(H_vals, N_vals, squared_loss):
        print(f"  H={h:7.2f} (round {round(h)})  N={n:10.1f}  squared_loss={s:.4f}")

    guide_img = Image.open(FIG_PATH).convert("RGB").copy()
    draw = ImageDraw.Draw(guide_img)
    t_left, t_top, t_right, t_bottom = crop_box_top
    for px in bottom_px:
        draw.line([(px, t_top), (px, t_bottom)], fill=(255, 0, 0), width=1)
    os.makedirs(os.path.dirname(GUIDE_IMAGE_PATH), exist_ok=True)
    guide_img.save(GUIDE_IMAGE_PATH)
    print(f"\nWrote {GUIDE_IMAGE_PATH}")

    top_clicks = click_markers(
        np.array(guide_img), crop_box_top,
        f"TOP panel (zero-one loss) -- click all {N_MARKERS} markers, guided by red lines, left to right",
    )

    top_px = np.array([p[0] for p in top_clicks])
    top_py = np.array([p[1] for p in top_clicks])
    order = np.argsort(top_px)
    top_px, top_py = top_px[order], top_py[order]

    top_N_vals = 10 ** (x_fit[0] * top_px + x_fit[1]) * 1000
    top_H_vals = (top_N_vals - 10) / 795
    zeroone_pct = y_fit_top[0] * top_py + y_fit_top[1]

    print("\nCross-check: H from top-panel clicks vs. bottom-panel clicks (should be close if you followed the guide lines):")
    any_mismatch = False
    for h_top, h_bot in zip(top_H_vals, H_vals):
        mismatch = abs(h_top - h_bot) >= 2
        any_mismatch = any_mismatch or mismatch
        flag = "  <-- mismatch, consider redoing this marker" if mismatch else ""
        print(f"  top H={h_top:7.2f}   bottom H={h_bot:7.2f}{flag}")
    if not any_mismatch:
        print("  All within tolerance.")

    orig_img = Image.open(FIG_PATH).convert("RGB")
    bottom_review = draw_crosses(orig_img, zip(bottom_px, bottom_py))
    bottom_review = crop_with_margin(bottom_review, crop_box_bottom)
    bottom_review.save(BOTTOM_REVIEW_PATH)

    top_review = draw_crosses(guide_img, zip(top_px, top_py))
    top_review = crop_with_margin(top_review, crop_box_top)
    top_review.save(TOP_REVIEW_PATH)

    print(f"\nWrote {BOTTOM_REVIEW_PATH} and {TOP_REVIEW_PATH} -- small red crosses mark exactly where you clicked.")
    print("Displaying both now -- close the window when done reviewing.")

    fig, (ax_b, ax_t) = plt.subplots(2, 1, figsize=(9, 12))
    ax_b.imshow(np.array(bottom_review))
    ax_b.set_title("Bottom panel (squared loss) -- your clicks", fontsize=9)
    ax_b.axis("off")
    ax_t.imshow(np.array(top_review))
    ax_t.set_title("Top panel (zero-one loss) -- your clicks (red guide lines still visible)", fontsize=9)
    ax_t.axis("off")
    fig.tight_layout()
    plt.show(block=True)

    proceed = input("\nDo the crosses land on the actual diamond markers? Save digitized CSV? [Y/n]: ").strip().lower()
    if proceed == "n":
        print(
            "Not saving. Bottom-panel clicks remain cached, so rerun this script and "
            "choose to reuse them if only the top panel needs redoing."
        )
        return

    # N is exactly determined by integer H (num_params(H) = 795*H + 10), so recompute it
    # from the rounded H rather than reporting the raw continuous pixel-position estimate --
    # N is a parameter count, it can't be fractional.
    H_rounded = [round(h) for h in H_vals]
    digitized = pd.DataFrame({
        "H": H_rounded,
        "N": [num_params(h) for h in H_rounded],
        "test_zeroone_pct": zeroone_pct,
        "test_squared_loss": squared_loss,
    })
    digitized.to_csv(DIGITIZED_PATH, index=False, float_format="%.4f")
    print(f"\nWrote {DIGITIZED_PATH}")
    print(f"\nH_VALS candidate (paste into config.py after checking the cross-check above):\n{digitized['H'].tolist()}")


if __name__ == "__main__":
    main()
