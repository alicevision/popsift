#!/usr/bin/env python3
"""Interactive view of matched keypoints that differ, per image of a report.

    inspect_matches.py <report dir>

<report dir> is a report_<a>__vs__<b> folder written by popsift_compare.py or
compare_paramsets.py. The window shows one input image. Every matched pair
whose difference exceeds the slider value is drawn with both glyphs (run a
in orange, run b in blue) and a line joining their centres.

    slider          threshold: pixels, degrees or descriptor distance (L2)
    mode buttons    pixel distance / rotation / descriptor distance
    < and >         previous / next image (also the left and right arrow keys)
    click           prints the values of the nearest drawn pair

The matched pairs come from matched.csv, or from matches.npz in the
explorer package.

Needs numpy and matplotlib (Testing/analysis/requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, RadioButtons, Slider

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sift_plots as sp  # noqa: E402

# Mode label: the matched.csv column, slider range and start value, and how
# the title names the threshold.
MODES = {
    "pixel dist. (px)": ("pos_dist", 0.0, 5.0, 0.5, "{:.2f} px in pixel distance"),
    "rotation (deg)": ("ori_diff_deg", 0.0, 180.0, 2.0,
                       "{:.1f} deg in rotation"),
    "descr. dist. (L2)": ("desc_dist", 0.0, 200.0, 20.0,
                          "{:.1f} in descriptor distance (L2)"),
}


class Viewer:
    def __init__(self, report_dir, index=0, mode=None):
        with open(os.path.join(report_dir, "summary.json"),
                  encoding="utf-8") as f:
            self.summary = json.load(f)
        self.report_dir = report_dir
        self.images = self.summary["images"]
        if not self.images:
            raise SystemExit(f"{report_dir}: the report has no images")
        self.index = index
        self.mode = mode if mode in MODES else next(iter(MODES))
        self.drawn = None  # indices of the pairs currently drawn, for clicks

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_axes([0.05, 0.16, 0.9, 0.78])
        _, lo, hi, init, _ = MODES[self.mode]
        self.slider = Slider(self.fig.add_axes([0.42, 0.06, 0.45, 0.03]),
                             "threshold", lo, hi, valinit=init)
        self.radio = RadioButtons(self.fig.add_axes([0.02, 0.01, 0.22, 0.12]),
                                  list(MODES), active=list(MODES).index(self.mode))
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
        self.m = sp.read_matches(self.path(entry["match_dir"]))
        self.draw()

    def set_mode(self, label):
        self.mode = label
        _, lo, hi, init, _ = MODES[label]
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

    def values(self):
        key = MODES[self.mode][0]
        v = self.m[key]
        return np.abs(v) if key == "ori_diff_deg" else v

    def draw(self):
        ax = self.ax
        ax.clear()
        sp.show_image(ax, self.image)
        thr = self.slider.val
        m = self.m
        self.drawn = np.nonzero(self.values() > thr)[0]
        d = self.drawn
        if len(d):
            for side, color in (("b", sp.COLOR_B), ("a", sp.COLOR_A)):
                sp.glyphs(ax, m[f"{side}_x"][d], m[f"{side}_y"][d],
                          m[f"{side}_sigma"][d], m[f"{side}_ori_deg"][d],
                          color)
            ax.add_collection(LineCollection(
                np.stack([np.stack([m["a_x"][d], m["a_y"][d]], axis=1),
                          np.stack([m["b_x"][d], m["b_y"][d]], axis=1)],
                         axis=1),
                colors="white", linewidths=0.8))
        entry = self.images[self.index]
        ax.set_title(f"[{self.index + 1}/{len(self.images)}] {entry['name']}: "
                     f"{len(d)} of {len(m['a_x'])} matched pairs differ by "
                     f"more than {MODES[self.mode][4].format(thr)}\n"
                     f"{self.summary['run_a']} (orange) vs "
                     f"{self.summary['run_b']} (blue)", fontsize=10)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes is not self.ax or self.drawn is None \
                or not len(self.drawn):
            return
        m, d = self.m, self.drawn
        best = d[np.argmin(np.hypot(m["a_x"][d] - event.xdata,
                                    m["a_y"][d] - event.ydata))]
        print(", ".join(f"{k}={m[k][best]:g}" for k in m), flush=True)


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
