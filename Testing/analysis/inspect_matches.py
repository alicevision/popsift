#!/usr/bin/env python3
"""Interactive view of matched keypoints that differ, per image of a report.

    inspect_matches.py <report dir>

<report dir> is a report_<a>__vs__<b> folder written by popsift_compare.py or
compare_paramsets.py. The window shows one input image. Every matched pair
whose difference exceeds the slider value is drawn with both glyphs (run a
in orange, run b in blue) and a line joining their centres.

    slider          threshold: pixels, degrees or descriptor distance
    mode buttons    position / rotation / descriptor difference
    < and >         previous / next image (also the left and right arrow keys)
    click           prints the matched.csv row of the nearest drawn pair

Needs numpy and matplotlib (Testing/analysis/requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sift_plots as sp  # noqa: E402

MODES = {
    "position (px)": ("pos_dist", 0.0, 5.0, 0.5),
    "rotation (deg)": ("ori_diff_deg", 0.0, 180.0, 2.0),
    "descriptor dist.": ("desc_dist", 0.0, 200.0, 20.0),
}
UNITS = {"pos_dist": " px", "ori_diff_deg": " deg", "desc_dist": ""}


class Viewer:
    def __init__(self, report_dir):
        with open(os.path.join(report_dir, "summary.json"),
                  encoding="utf-8") as f:
            self.summary = json.load(f)
        self.report_dir = report_dir
        self.images = self.summary["images"]
        if not self.images:
            raise SystemExit(f"{report_dir}: the report has no images")
        self.index = 0
        self.mode = next(iter(MODES))
        self.drawn = []  # rows currently drawn, for clicks

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_axes([0.05, 0.16, 0.9, 0.78])
        key, lo, hi, init = MODES[self.mode]
        self.slider = Slider(self.fig.add_axes([0.42, 0.06, 0.45, 0.03]),
                             "threshold", lo, hi, valinit=init)
        self.radio = RadioButtons(self.fig.add_axes([0.02, 0.01, 0.22, 0.12]),
                                  list(MODES))
        self.prev_btn = Button(self.fig.add_axes([0.42, 0.01, 0.06, 0.04]), "<")
        self.next_btn = Button(self.fig.add_axes([0.49, 0.01, 0.06, 0.04]), ">")

        self.slider.on_changed(lambda _: self.draw())
        self.radio.on_clicked(self.set_mode)
        self.prev_btn.on_clicked(lambda _: self.step(-1))
        self.next_btn.on_clicked(lambda _: self.step(+1))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.load()

    def path(self, rel):
        return os.path.normpath(os.path.join(self.report_dir, rel))

    def load(self):
        entry = self.images[self.index]
        self.image = sp.read_pgm(self.path(entry["input"]))
        self.rows = sp.read_rows(os.path.join(self.path(entry["match_dir"]),
                                              "matched.csv"))
        self.draw()

    def set_mode(self, label):
        self.mode = label
        _, lo, hi, init = MODES[label]
        self.slider.valmin, self.slider.valmax = lo, hi
        self.slider.ax.set_xlim(lo, hi)
        self.slider.set_val(init)  # redraws

    def step(self, delta):
        self.index = (self.index + delta) % len(self.images)
        self.load()

    def on_key(self, event):
        if event.key == "right":
            self.step(+1)
        elif event.key == "left":
            self.step(-1)

    def value(self, row):
        key = MODES[self.mode][0]
        v = float(row[key])
        return abs(v) if key == "ori_diff_deg" else v

    def draw(self):
        ax = self.ax
        ax.clear()
        sp.show_image(ax, self.image)
        thr = self.slider.val
        self.drawn = [r for r in self.rows if self.value(r) > thr]
        if self.drawn:
            for side, color in (("b", sp.COLOR_B), ("a", sp.COLOR_A)):
                sp.glyphs(ax, [float(r[f"{side}_x"]) for r in self.drawn],
                          [float(r[f"{side}_y"]) for r in self.drawn],
                          [float(r[f"{side}_sigma"]) for r in self.drawn],
                          [float(r[f"{side}_ori_deg"]) for r in self.drawn],
                          color)
            for r in self.drawn:
                ax.plot([float(r["a_x"]), float(r["b_x"])],
                        [float(r["a_y"]), float(r["b_y"])],
                        color="white", linewidth=0.8)
        entry = self.images[self.index]
        key = MODES[self.mode][0]
        unit = UNITS[key]
        ax.set_title(f"[{self.index + 1}/{len(self.images)}] {entry['name']}: "
                     f"{len(self.drawn)} of {len(self.rows)} matched pairs "
                     f"differ by more than {thr:.2f}{unit} in {key}\n"
                     f"{self.summary['run_a']} (orange) vs "
                     f"{self.summary['run_b']} (blue)", fontsize=10)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes is not self.ax or not self.drawn:
            return
        best = min(self.drawn, key=lambda r: math.hypot(
            float(r["a_x"]) - event.xdata, float(r["a_y"]) - event.ydata))
        print(", ".join(f"{k}={v}" for k, v in best.items()), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    parser.add_argument("report_dir", help="a report_<a>__vs__<b> folder")
    args = parser.parse_args(argv)
    Viewer(args.report_dir)
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
