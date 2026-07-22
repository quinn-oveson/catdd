"""Digitize Belkin's Fig. 3 Train curves (train zero-one loss %, train squared
loss) by clicking the curve at each of our already-established H_VALS
x-positions.

Unlike the Test curve (diamond markers, see extract_belkin_markers.py), the
Train curve is a plain line with no markers -- nothing to click "on" in the
way a marker gives you a discrete target, but also no ambiguity about which
of several crowded points you're reading, since it's one continuous line at
each x. We already know exactly which H values we care about (config.H_VALS,
the corrected digitized list) and therefore exactly which pixel column each
one sits at, by inverting calibrate_belkin_figure.py's x_fit -- so this script
draws a vertical guide line at each of those 23 known x-positions and has you
click where that line crosses the orange Train curve. Only the y-coordinate
of each click is used; x is fixed by construction, so there's no cross-panel
H reconciliation needed like there was for the Test markers.

Run this LOCALLY (needs a real display, not over SSH), after both
calibrate_belkin_figure.py and extract_belkin_markers.py have been run --
this script only adds train_zeroone_pct / train_squared_loss columns to the
existing results/belkin_digitized.csv (reusing its H/N columns), it doesn't
create that file.

As before, click order is top panel first, its clicks cached to
results/belkin_train_top_clicks.json so a bad bottom-panel pass doesn't force
you to redo the top one. Between clicks you can pan/zoom with the matplotlib
toolbar -- just toggle the zoom/pan tool off again before clicking, or the
click will pan/zoom instead of registering as a point.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from utils import num_params

CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "belkin_calibration.json")
FIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "belkin_figure3.png")
DIGITIZED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "belkin_digitized.csv")
TOP_CLICKS_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "belkin_train_top_clicks.json")
TOP_REVIEW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "belkin_train_top_panel_review.png")
BOTTOM_REVIEW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "belkin_train_bottom_panel_review.png")

GUIDE_COLOR = (0, 200, 0)  # distinct from the blue Test curve and orange Train curve
CROSS_COLOR = (255, 0, 0)
CROSS_SIZE = 6


def px_for_H(H, x_fit):
    value_x1000 = num_params(H) / 1000
    return (np.log10(value_x1000) - x_fit[1]) / x_fit[0]


def draw_guides(base_img, px_list, y0, y1, color=GUIDE_COLOR):
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    for px in px_list:
        draw.line([(px, y0), (px, y1)], fill=color, width=1)
    return img


def draw_crosses(base_img, points, size=CROSS_SIZE, color=CROSS_COLOR):
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    for px, py in points:
        draw.line([(px - size, py), (px + size, py)], fill=color, width=1)
        draw.line([(px, py - size), (px, py + size)], fill=color, width=1)
    return img


def crop_with_margin(img, box, margin=15):
    left, top, right, bottom = box
    w, h = img.size
    return img.crop((max(0, left - margin), max(0, top - margin), min(w, right + margin), min(h, bottom + margin)))


def click_along_guides(img, crop_box, H_list, title):
    left, top, right, bottom = crop_box
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img)
    ax.set_xlim(left - 15, right + 15)
    ax.set_ylim(bottom + 15, top - 15)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()

    points = []
    print(f"\n{title}")
    for i, H in enumerate(H_list):
        print(f"  Click where the H={H} guide line crosses the Train curve ({i + 1}/{len(H_list)})...")
        pt = plt.ginput(1, timeout=0)
        while not pt:
            print("  No click registered -- try again.")
            pt = plt.ginput(1, timeout=0)
        points.append(pt[0])
    plt.close(fig)
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--redo_H", type=str, default=None,
        help="Comma-separated list of H values to re-click (e.g. '100,200,300,1000'), "
             "leaving all other rows in belkin_digitized.csv untouched. If omitted, "
             "redoes all H_VALS from scratch (the original full workflow).",
    )
    args = parser.parse_args()

    if not os.path.exists(CALIBRATION_PATH):
        raise SystemExit(f"{CALIBRATION_PATH} not found -- run calibrate_belkin_figure.py first.")
    if not os.path.exists(DIGITIZED_PATH):
        raise SystemExit(f"{DIGITIZED_PATH} not found -- run extract_belkin_markers.py first (need its H/N columns).")

    with open(CALIBRATION_PATH) as f:
        cal = json.load(f)
    x_fit = cal["x_fit"]
    y_fit_top = cal["y_fit_top"]
    y_fit_bottom = cal["y_fit_bottom"]
    crop_box_top = cal["crop_box_top"]
    crop_box_bottom = cal["crop_box_bottom"]

    digitized = pd.read_csv(DIGITIZED_PATH)
    all_H = digitized["H"].tolist()

    redoing_subset = args.redo_H is not None
    if redoing_subset:
        redo_set = {int(h) for h in args.redo_H.split(",")}
        missing = redo_set - set(all_H)
        if missing:
            raise SystemExit(f"H values {missing} not found in {DIGITIZED_PATH} (have: {all_H})")
        H_list = [h for h in all_H if h in redo_set]
        print(f"Redoing only H={H_list}; all other rows will be left untouched.")
    else:
        H_list = all_H

    px_list = [px_for_H(h, x_fit) for h in H_list]

    orig_img = Image.open(FIG_PATH).convert("RGB")
    t_left, t_top, t_right, t_bottom = crop_box_top
    b_left, b_top, b_right, b_bottom = crop_box_bottom

    top_guide_img = draw_guides(orig_img, px_list, t_top, t_bottom)
    bottom_guide_img = draw_guides(orig_img, px_list, b_top, b_bottom)

    top_clicks = None
    if not redoing_subset and os.path.exists(TOP_CLICKS_CACHE):
        reuse = input(
            f"Found cached top-panel clicks at {TOP_CLICKS_CACHE}. Reuse them "
            "and only redo the bottom panel? [Y/n]: "
        ).strip().lower()
        if reuse != "n":
            with open(TOP_CLICKS_CACHE) as f:
                top_clicks = json.load(f)

    if top_clicks is None:
        top_clicks = click_along_guides(
            np.array(top_guide_img), crop_box_top, H_list,
            f"TOP panel (zero-one loss) -- click Train curve at each of {len(H_list)} guide lines",
        )
        if not redoing_subset:
            with open(TOP_CLICKS_CACHE, "w") as f:
                json.dump(top_clicks, f)

    bottom_clicks = click_along_guides(
        np.array(bottom_guide_img), crop_box_bottom, H_list,
        f"BOTTOM panel (squared loss) -- click Train curve at each of {len(H_list)} guide lines",
    )

    top_py = np.array([p[1] for p in top_clicks])
    bottom_py = np.array([p[1] for p in bottom_clicks])
    train_zeroone_pct = y_fit_top[0] * top_py + y_fit_top[1]
    train_squared_loss = y_fit_bottom[0] * bottom_py + y_fit_bottom[1]

    top_review = draw_crosses(top_guide_img, zip([p[0] for p in top_clicks], top_py))
    top_review = crop_with_margin(top_review, crop_box_top)
    top_review.save(TOP_REVIEW_PATH)

    bottom_review = draw_crosses(bottom_guide_img, zip([p[0] for p in bottom_clicks], bottom_py))
    bottom_review = crop_with_margin(bottom_review, crop_box_bottom)
    bottom_review.save(BOTTOM_REVIEW_PATH)

    print(f"\nWrote {TOP_REVIEW_PATH} and {BOTTOM_REVIEW_PATH} -- magenta crosses mark exactly where you clicked.")
    print("Displaying both now -- close the window when done reviewing.")

    fig, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(9, 12))
    ax_t.imshow(np.array(top_review))
    ax_t.set_title("Top panel (zero-one loss) -- Train curve clicks", fontsize=9)
    ax_t.axis("off")
    ax_b.imshow(np.array(bottom_review))
    ax_b.set_title("Bottom panel (squared loss) -- Train curve clicks", fontsize=9)
    ax_b.axis("off")
    fig.tight_layout()
    plt.show(block=True)

    proceed = input(
        "\nDo the crosses land on the Train curve at each guide line? "
        "Save train_zeroone_pct / train_squared_loss into belkin_digitized.csv? [Y/n]: "
    ).strip().lower()
    if proceed == "n":
        print(
            "Not saving. Top-panel clicks remain cached, so rerun this script and "
            "choose to reuse them if only the bottom panel needs redoing."
        )
        return

    if redoing_subset:
        mask = digitized["H"].isin(redo_set)
        digitized.loc[mask, "train_zeroone_pct"] = train_zeroone_pct
        digitized.loc[mask, "train_squared_loss"] = train_squared_loss
    else:
        digitized["train_zeroone_pct"] = train_zeroone_pct
        digitized["train_squared_loss"] = train_squared_loss

    digitized.to_csv(DIGITIZED_PATH, index=False, float_format="%.4f")
    print(f"\nUpdated {DIGITIZED_PATH} with train_zeroone_pct and train_squared_loss columns.")


if __name__ == "__main__":
    main()
