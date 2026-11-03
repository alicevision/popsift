#!/usr/bin/env python3
"""Explore the comparison of PopSift with VLFeat, image by image.

    explorer.py [<data dir>]

<data dir> holds report_<a>__vs__<b> folders: the data/ folder of an explorer
package (the default when explorer.py sits in one), or the work dir of a
comparison run, e.g. <build>/Testing/compareVlfeat/run.

    left            image sets (one per report) and their images; click to
                    select, up/down arrow keys step through the images
    tabs            Image: key figures and distributions of one image
                    Rotation vs descriptor: how a rotation difference
                    affects the descriptor distance
                    Sorted: the sorted differences of the image and the set
                    Heatmaps: differences by place in the image and by
                    descriptor bin (these two use the right column too)
                    Set: key figures of the whole set
                    Glossary: what the terms mean
    right           keypoints that only one side found; the button opens
                    the match inspector for the image in its own window

Needs numpy and matplotlib (requirements.txt). glossary.md explains the terms.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import inspect_matches as im  # noqa: E402
import sift_plots as sp  # noqa: E402

TABS = ("Image", "Rotation vs descriptor", "Sorted", "Heatmaps", "Set",
        "Glossary")
# Tabs that use the right column too.
WIDE_TABS = ("Sorted", "Heatmaps")
BACKGROUND = "#f4f4f2"
SELECTED = "#dbe9f7"
MUTED = "#666666"


def run_name(run):
    """How a run is called for readers."""
    return "VLFeat" if run == "vlfeat" else "PopSift"


def fmt(x, digits=3, unit=""):
    return "-" if x is None else f"{x:.{digits}f}{unit}"


def share(n, of):
    """n and its share of 'of', e.g. "74894 (99.1%)"."""
    if n is None:
        return "-"
    return f"{n} ({100 * n / of:.1f}%)" if of else f"{n}"


def matched_pct(stats):
    """Matched keypoints as a share of the keypoints of a."""
    c = stats["counts"]
    return f"{100 * c['matched'] / c['a']:.1f}%" if c["a"] else "-"


class Report:
    """One report_<a>__vs__<b> folder."""

    def __init__(self, path):
        self.dir = path
        with open(os.path.join(path, "summary.json"), encoding="utf-8") as f:
            self.summary = json.load(f)
        self.images = self.summary["images"]
        self.set = self.summary["set"]
        self.run_a = self.summary["run_a"]
        self.run_b = self.summary["run_b"]

    def path(self, rel):
        return os.path.normpath(os.path.join(self.dir, rel))

    def label(self, with_runs):
        return f"{self.set} ({self.run_a} vs {self.run_b})" if with_runs \
            else self.set


def find_reports(data_dir):
    paths = sorted(os.path.dirname(p) for p in glob.glob(
        os.path.join(data_dir, "**", "report_*", "summary.json"),
        recursive=True))
    return [Report(p) for p in paths]


def glossary_text():
    path = os.path.join(HERE, "glossary.md")
    if not os.path.exists(path):
        return "glossary.md is missing."
    with open(path, encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"^# .*\n+", "", text)
    return text.replace("**", "").replace("`", "")


class Explorer:
    def __init__(self, data_dir):
        self.reports = find_reports(data_dir)
        if not self.reports:
            raise SystemExit(f"no report_*/summary.json below {data_dir}")
        self.several_pairs = len({(r.run_a, r.run_b)
                                  for r in self.reports}) > 1
        self.r = 0          # selected report
        self.i = 0          # selected image
        self.tab = TABS[0]
        self.content = []   # axes of the current tab
        self.viewers = []   # open inspectors, kept from garbage collection
        self.matches = None

        self.fig = plt.figure(figsize=(16, 10), facecolor=BACKGROUND)
        self.fig.canvas.manager.set_window_title("PopSift vs VLFeat explorer")
        self.header = self.fig.text(0.01, 0.965, "", fontsize=15,
                                    weight="bold")
        self.subheader = self.fig.text(0.01, 0.94, "", fontsize=10,
                                       color=MUTED)
        self.ax_sets = self.panel([0.01, 0.50, 0.19, 0.42])
        self.ax_images = self.panel([0.01, 0.03, 0.19, 0.45])
        self.tab_buttons = []
        for k, name in enumerate(TABS):
            b = Button(self.fig.add_axes([0.215 + k * 0.129, 0.885, 0.124,
                                          0.035]), name)
            b.on_clicked(lambda _, n=name: self.set_tab(n))
            self.tab_buttons.append(b)
        self.ax_unmatched = self.fig.add_axes([0.655, 0.27, 0.335, 0.60])
        inspector_text = self.fig.text(
            0.655, 0.235, "Open the match inspector for pairs that differ in",
            fontsize=9)
        self.radio = RadioButtons(self.fig.add_axes([0.655, 0.10, 0.335,
                                                     0.12]),
                                  list(im.MODES), active=1)
        self.open_btn = Button(self.fig.add_axes([0.655, 0.03, 0.335, 0.055]),
                               "Open match inspector for this image",
                               color=SELECTED)
        self.open_btn.on_clicked(lambda _: self.open_viewer())
        self.right_column = [self.ax_unmatched, inspector_text,
                             self.radio.ax, self.open_btn.ax]
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.select(0, 0)

    def panel(self, rect):
        ax = self.fig.add_axes(rect)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        return ax

    # --- selection ------------------------------------------------------------

    @property
    def report(self):
        return self.reports[self.r]

    @property
    def entry(self):
        return self.report.images[self.i]

    def select(self, r, i):
        self.r = r % len(self.reports)
        self.i = i % len(self.report.images)
        self.matches = sp.read_matches(self.report.path(self.entry["match_dir"]))
        self.draw()

    def on_click(self, event):
        for ax, count, pick in (
                (self.ax_sets, len(self.reports),
                 lambda k: self.select(k, 0)),
                (self.ax_images, len(self.report.images),
                 lambda k: self.select(self.r, k))):
            if event.inaxes is ax and event.ydata is not None:
                k = int((0.9 - event.ydata) / self.row_height(count) + 0.5)
                if 0 <= k < count:
                    pick(k)

    def on_key(self, event):
        if event.key == "down":
            self.select(self.r, self.i + 1)
        elif event.key == "up":
            self.select(self.r, self.i - 1)
        elif event.key == "pagedown":
            self.select(self.r + 1, 0)
        elif event.key == "pageup":
            self.select(self.r - 1, 0)

    def set_tab(self, name):
        self.tab = name
        self.draw()

    def open_viewer(self):
        viewer = im.Viewer(self.report.dir, self.i, self.radio.value_selected)
        self.viewers.append(viewer)
        viewer.fig.show()

    # --- drawing --------------------------------------------------------------

    @staticmethod
    def row_height(count):
        return min(0.08, 0.85 / max(count, 1))

    def draw_list(self, ax, title, columns, rows, selected):
        ax.clear()
        self.panel_style(ax)
        h = self.row_height(len(rows))
        ax.text(0.03, 0.96, title, fontsize=12, weight="bold", va="center")
        ax.text(0.03, 0.915, columns, fontsize=8, color=MUTED,
                family="monospace", va="center")
        for k, row in enumerate(rows):
            y = 0.9 - (k + 0.5) * h
            if k == selected:
                ax.axhspan(y - h / 2, y + h / 2, 0.01, 0.99, color=SELECTED)
            ax.text(0.03, y, row, fontsize=min(8.5, 300 * h), va="center",
                    family="monospace")

    @staticmethod
    def panel_style(ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

    def draw(self):
        rep = self.report
        self.header.set_text("PopSift vs VLFeat explorer")
        self.subheader.set_text(
            f"PopSift parameter set: {rep.run_a}   ·   reference: "
            f"{rep.run_b}   ·   {len(self.reports)} image set(s)   ·   "
            "descriptor distances are L2, in byte units")
        self.draw_list(
            self.ax_sets, "Image sets", "set              images  matched",
            [f"{r.label(self.several_pairs)[:16]:<16} {len(r.images):>5}  "
             f"{matched_pct(r.summary['set_stats']):>7}"
             for r in self.reports], self.r)
        self.draw_list(
            self.ax_images, f"Images of {rep.set}",
            f"image        {run_name(rep.run_a)[:7]:>7} "
            f"{run_name(rep.run_b)[:7]:>7}  matched",
            [f"{e['name'][:12]:<12} {e['counts']['a']:>7} "
             f"{e['counts']['b']:>7}  {matched_pct(e):>7}"
             for e in rep.images], self.i)
        for b, name in zip(self.tab_buttons, TABS):
            b.color = "#2b2b2b" if name == self.tab else "white"
            b.hovercolor = b.color
            b.ax.set_facecolor(b.color)
            b.label.set_color("white" if name == self.tab else "#333333")
        for ax in self.content:
            self.fig.delaxes(ax)
        self.content = []
        {"Image": self.draw_image_tab,
         "Rotation vs descriptor": self.draw_scatter_tab,
         "Sorted": self.draw_sorted_tab,
         "Heatmaps": self.draw_heatmaps_tab,
         "Set": self.draw_set_tab,
         "Glossary": self.draw_glossary_tab}[self.tab]()
        for artist in self.right_column:
            artist.set_visible(self.tab not in WIDE_TABS)
        if self.tab not in WIDE_TABS:
            self.draw_unmatched()
        self.fig.canvas.draw_idle()

    def add(self, rect, **kwargs):
        ax = self.fig.add_axes(rect, **kwargs)
        self.content.append(ax)
        return ax

    def tiles(self, ax, title, stats, top=0.95):
        rep = self.report
        a, b = run_name(rep.run_a), run_name(rep.run_b)
        c = stats["counts"]
        m = stats["metrics"]
        values = [
            (f"keypoints {a}", f"{c['a']}"),
            (f"keypoints {b}", f"{c['b']}"),
            (f"matched, share of {a}", share(c["matched"], c["a"])),
            ("nearest descriptor in place", share(c.get("nn_in_place_a"),
                                                  c["a"])),
            ("... and passes Lowe's 0.8 ratio test",
             share(c.get("nn_distinctive_a"), c["a"])),
            ("nearest-descriptor agreement", fmt(stats["nn_agreement_a"])),
            ("pixel distance, median", fmt(m["pos_dist_px"]["median"], 3,
                                           " px")),
            ("rotation difference, median",
             fmt(m["abs_ori_diff_deg"]["median"], 2, "°")),
            ("descriptor distance (L2), median",
             fmt(m["desc_dist_l2"]["median"], 2)),
        ]
        ax.text(0.03, top, title, fontsize=12, weight="bold", va="top")
        ax.text(0.03, top - 0.09,
                f"matched: a {b} keypoint at most max(1 px, 0.5 sigma) "
                f"away (the matching radius), with a scale\n"
                f"   within a factor 1.5; paired one to one, closest first\n"
                f"nearest descriptor in place: the most similar {b} "
                f"descriptor belongs to a keypoint\n"
                f"   within the matching radius; distinctive: it also passes "
                f"Lowe's ratio test (< 0.8 × second)\n"
                f"nearest-descriptor agreement: share of the matched {a} "
                f"keypoints whose most similar\n"
                f"   {b} descriptor is their matched keypoint",
                fontsize=7.5, color=MUTED, va="top", linespacing=1.3)
        for k, (label, value) in enumerate(values):
            x = 0.03 + (k % 3) * 0.33
            y = top - 0.5 - (k // 3) * 0.17
            ax.text(x, y + 0.055, label, fontsize=8, color=MUTED)
            ax.text(x, y - 0.025, value, fontsize=13, weight="bold")

    def draw_image_tab(self):
        e = self.entry
        box = self.add([0.215, 0.47, 0.43, 0.40])
        self.panel_style(box)
        self.tiles(box, f"{self.report.set} / {e['name']}", e)
        panel = self.add([0.215, 0.03, 0.43, 0.42])
        self.panel_style(panel)
        panel.text(0.03, 0.97, "How close are the matched keypoints?",
                   fontsize=11, weight="bold", va="top")
        m = self.matches
        for k, (values, label, bottom, top) in enumerate((
                (np.abs(m["ori_diff_deg"]), "rotation difference (deg)",
                 0.01, 180.0),
                (m["desc_dist"], "descriptor distance (L2)", 1.0,
                 1.2 * max(m["desc_dist"], default=1.0)))):
            ax = self.add([0.255 + k * 0.205, 0.09, 0.165, 0.30])
            v = np.sort(values)
            if len(v):
                # Values outside the axis are drawn at its ends.
                ax.plot(np.clip(v, bottom, top), np.arange(1, len(v) + 1) / len(v),
                        color="#333333")
                for q in (0.5, 0.9, 0.99):
                    x = float(np.quantile(v, q))
                    ax.axhline(q, color="#bbbbbb", linewidth=0.6)
                    ax.annotate(f"{int(q * 100)}%: {x:.2f}",
                                (min(max(x, bottom), top), q), xytext=(4, -10),
                                textcoords="offset points", fontsize=7)
            sp.log_axis(ax, "x", bottom, top)
            ax.set_ylim(0, 1.01)
            ax.set_xlabel(f"{label}, below {bottom:g} at {bottom:g}")
            ax.grid(axis="y", alpha=0.3)
            if k == 0:
                ax.set_ylabel("share of matched keypoints")

    def draw_scatter_tab(self):
        m = self.matches
        ax = self.add([0.26, 0.09, 0.37, 0.74])
        rot = np.abs(m["ori_diff_deg"])
        same = m["pos_dist"] <= 0.25
        # Values outside the axes are drawn at their ends.
        x = np.clip(rot, 0.01, 180.0)
        y = np.maximum(m["desc_dist"], 1.0)
        ax.scatter(x[same], y[same], s=4, alpha=0.3,
                   color=sp.COLOR_B, linewidths=0,
                   label="at the same position (≤ 0.25 px)")
        ax.scatter(x[~same], y[~same], s=6, alpha=0.6,
                   color=sp.COLOR_A, linewidths=0,
                   label="more than 0.25 px apart")
        sp.log_axis(ax, "x", 0.01, 180.0)
        sp.log_axis(ax, "y", 1.0, 1.2 * y.max() if len(rot) else 10.0)
        ax.set_xlabel("rotation difference (deg), below 0.01 at 0.01")
        ax.set_ylabel("descriptor distance (L2), below 1 at 1")
        ax.legend(fontsize=8, loc="upper left")
        if len(rot) > 1:
            r = np.corrcoef(np.log1p(rot), np.log1p(m["desc_dist"]))[0, 1]
            note = f"correlation of the logarithms: {r:.2f}"
        else:
            note = ""
        ax.set_title(f"{self.report.set} / {self.entry['name']}: "
                     f"{len(rot)} matched pairs\nA rotated keypoint has a "
                     f"rotated descriptor. {note}", fontsize=10)

    def show_png(self, rect, path):
        ax = self.add(rect)
        ax.set_xticks([])
        ax.set_yticks([])
        if os.path.exists(path):
            ax.imshow(plt.imread(path))
        else:
            ax.set_facecolor("white")
            ax.text(0.5, 0.5, f"no {os.path.basename(path)} here; it comes "
                    "with the plots of a newer comparison run", ha="center",
                    transform=ax.transAxes)
        return ax

    def draw_sorted_tab(self):
        rep = self.report
        self.show_png([0.215, 0.455, 0.775, 0.415],
                      os.path.join(rep.path(self.entry["match_dir"]),
                                   "sorted.png"))
        self.show_png([0.215, 0.03, 0.775, 0.415],
                      os.path.join(rep.dir, "sorted.png"))

    def draw_heatmaps_tab(self):
        match_dir = self.report.path(self.entry["match_dir"])
        self.show_png([0.215, 0.44, 0.775, 0.43],
                      os.path.join(match_dir, "image_heatmap.png"))
        self.show_png([0.215, 0.03, 0.775, 0.40],
                      os.path.join(match_dir, "descriptor_bins.png"))

    def draw_set_tab(self):
        rep = self.report
        box = self.add([0.215, 0.47, 0.43, 0.40])
        self.panel_style(box)
        stats = rep.summary["set_stats"]
        self.tiles(box, f"{rep.set}: {len(rep.images)} image(s)", stats)
        cdf = os.path.join(rep.dir, "cdf.png")
        ax = self.add([0.215, 0.03, 0.43, 0.42])
        ax.set_xticks([])
        ax.set_yticks([])
        if os.path.exists(cdf):
            ax.imshow(plt.imread(cdf))
        else:
            ax.text(0.5, 0.5, "no cdf.png in this report", ha="center")

    def draw_glossary_tab(self):
        ax = self.add([0.215, 0.03, 0.43, 0.84])
        self.panel_style(ax)
        ax.text(0.03, 0.98, glossary_text(), fontsize=7.5, va="top",
                wrap=True)

    def draw_unmatched(self):
        ax = self.ax_unmatched
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        rep, e = self.report, self.entry
        png = os.path.join(rep.path(e["match_dir"]), "unmatched.png")
        c = e["counts"]
        ax.set_title(f"Keypoints found by only one side: "
                     f"{c['only_a']} only {run_name(rep.run_a)} (orange), "
                     f"{c['only_b']} only {run_name(rep.run_b)} (blue)",
                     fontsize=9)
        if os.path.exists(png):
            ax.imshow(plt.imread(png))
        else:
            ax.text(0.5, 0.5, "no unmatched.png", ha="center",
                    transform=ax.transAxes)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    default = os.path.join(HERE, "data")
    parser.add_argument("data_dir", nargs="?", default=default,
                        help="folder with report_* folders (default: data/ "
                             "next to this script)")
    parser.add_argument("--save", metavar="PNG",
                        help="write the window to a PNG file and exit "
                             "(with --set, --image and --tab)")
    parser.add_argument("--set", type=int, default=0,
                        help="with --save: index of the image set")
    parser.add_argument("--image", type=int, default=0,
                        help="with --save: index of the image")
    parser.add_argument("--tab", choices=TABS, default=TABS[0],
                        help="with --save: the tab")
    args = parser.parse_args(argv)
    if args.save:
        matplotlib.use("Agg")
    explorer = Explorer(args.data_dir)
    if args.save:
        explorer.tab = args.tab
        explorer.select(args.set, args.image)
        explorer.fig.savefig(args.save, dpi=100,
                             facecolor=explorer.fig.get_facecolor())
        return 0
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
