"""Drawing SIFT keypoints over their input image, with numpy and matplotlib.

Used by Testing/scripts/popsift_compare.py and compare_paramsets.py
(unmatched.png and cdf.png) and by inspect_matches.py. Keypoints are drawn with the classic SIFT glyph: a
circle whose radius is proportional to the scale, and a line from the centre
in the direction of the orientation. Image coordinates have y pointing down,
as in both PopSift's and VLFeat's output.
"""

from __future__ import annotations

import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, FuncFormatter

# Colour-blind safe pair (Okabe-Ito): run a in orange, run b in blue.
COLOR_A = "#E69F00"
COLOR_B = "#0072B2"

# Colour-blind safe colours (Okabe-Ito) for several curves in one plot.
SERIES_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
                 "#D55E00", "#F0E442", "#000000"]

# Glyph radius in multiples of the keypoint's sigma.
GLYPH_SCALE = 3.0


def read_pgm(path):
    """Reads an 8-bit P5 or P2 PGM file into a 2D numpy array."""
    with open(path, "rb") as f:
        raw = f.read()
    tokens, pos = [], 0
    while len(tokens) < 4:
        while raw[pos:pos + 1].isspace():
            pos += 1
        if raw[pos:pos + 1] == b"#":
            while raw[pos:pos + 1] not in (b"\n", b"\r", b""):
                pos += 1
            continue
        start = pos
        while pos < len(raw) and not raw[pos:pos + 1].isspace():
            pos += 1
        tokens.append(raw[start:pos])
    magic = tokens[0]
    w, h, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    if magic == b"P5":
        dtype = np.uint8 if maxval < 256 else ">u2"
        data = np.frombuffer(raw, dtype=dtype, count=w * h, offset=pos + 1)
    elif magic == b"P2":
        data = np.array(raw[pos:].split()[:w * h], dtype=np.int32)
    else:
        raise ValueError(f"{path}: not a grey PGM ({magic!r})")
    return data.reshape(h, w)


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def glyphs(ax, xs, ys, sigmas, oris_deg, color, scale=GLYPH_SCALE, lw=1.0,
           picker=None):
    """Draws SIFT glyphs and returns the artists."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    r = np.asarray(sigmas, dtype=float) * scale
    t = np.radians(np.asarray(oris_deg, dtype=float))
    circles = PatchCollection([Circle((x, y), rr) for x, y, rr in zip(xs, ys, r)],
                              facecolor="none", edgecolor=color, linewidth=lw)
    ax.add_collection(circles)
    seg = np.stack([np.stack([xs, ys], axis=1),
                    np.stack([xs + r * np.cos(t), ys + r * np.sin(t)], axis=1)],
                   axis=1)
    lines = LineCollection(seg, colors=color, linewidths=lw)
    ax.add_collection(lines)
    return circles, lines


def show_image(ax, image):
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(image.shape[0] - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def figure_for(image, min_inches=5.0, max_inches=12.0):
    """A figure of 5 to 12 inches on the long side, so tiny images are
    enlarged and large ones fit on a screen."""
    h, w = image.shape
    inches = min(max(max(w, h) / 100.0, min_inches), max_inches)
    return plt.figure(figsize=(max(inches * w / max(w, h), 3),
                               max(inches * h / max(w, h), 3)))


def render_unmatched(pgm, match_dir, out_png, label_a="a", label_b="b"):
    """Writes an image of the keypoints that only one of the runs found."""
    image = read_pgm(pgm)
    only_a = read_rows(os.path.join(match_dir, "only_a.csv"))
    only_b = read_rows(os.path.join(match_dir, "only_b.csv"))
    fig = figure_for(image)
    ax = fig.add_axes([0, 0, 1, 0.92])
    show_image(ax, image)
    for rows, color in ((only_a, COLOR_A), (only_b, COLOR_B)):
        if rows:
            glyphs(ax, [float(r["x"]) for r in rows],
                   [float(r["y"]) for r in rows],
                   [float(r["sigma"]) for r in rows],
                   [float(r["ori_deg"]) for r in rows], color)
    fig.text(0.01, 0.99, f"only {label_a}: {len(only_a)}", color=COLOR_A,
             va="top", ha="left", fontsize=10, weight="bold")
    fig.text(0.99, 0.99, f"only {label_b}: {len(only_b)}", color=COLOR_B,
             va="top", ha="right", fontsize=10, weight="bold")
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png


def _ecdf(ax, values, **kwargs):
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0:
        return
    y = np.arange(1, v.size + 1) / v.size
    ax.step(np.concatenate([[v[0]], v]), np.concatenate([[0.0], y]),
            where="post", **kwargs)


def log_axis(ax, which, bottom, top):
    """Makes the x or y axis of ax logarithmic from bottom to top, with
    labels at every power of 10 and grid lines at 1 to 9 times every power
    of 10. Values outside must be drawn at the edges (np.clip)."""
    axis = ax.xaxis if which == "x" else ax.yaxis
    top = max(top, 10 * bottom)
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    (ax.set_xlim if which == "x" else ax.set_ylim)(bottom, top)
    decades = range(math.floor(math.log10(bottom)),
                    math.ceil(math.log10(top)) + 1)
    axis.set_major_locator(FixedLocator([10.0 ** k for k in decades]))
    axis.set_minor_locator(FixedLocator(
        [m * 10.0 ** k for k in decades for m in range(2, 10)]))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.grid(True, which="major", axis=which, alpha=0.5)
    ax.grid(True, which="minor", axis=which, alpha=0.2)


def render_cdfs(out_png, title, series, match_radius=1.0):
    """Writes cumulative distributions of descriptor and position distances.

    series is a list of (label, rows) where rows is match_stats.read_rows()
    output, possibly merged over images. Left: descriptor distance of the
    geometrically matched pairs, solid for pairs with the same pose, dashed
    for all. Right: position distance from each keypoint to the keypoint
    with the most similar descriptor in the other run, solid a in b, dashed
    b in a, on a logarithmic axis from 0.01 px (smaller distances are drawn
    at 0.01 px); the grey line is the smallest match radius.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, (label, rows) in enumerate(series):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        prefix = f"{label}: " if len(series) > 1 else ""
        _ecdf(left, [float(r["desc_dist"]) for r in rows["same_pose"]],
              color=color, label=f"{prefix}same pose ({len(rows['same_pose'])})")
        _ecdf(left, [float(r["desc_dist"]) for r in rows["matched"]],
              color=color, linestyle="--",
              label=f"{prefix}all matched ({len(rows['matched'])})")
        _ecdf(right, [max(float(r["pos_dist"]), 0.01) for r in rows["nn_a"]],
              color=color, label=f"{prefix}a in b ({len(rows['nn_a'])})")
        _ecdf(right, [max(float(r["pos_dist"]), 0.01) for r in rows["nn_b"]],
              color=color, linestyle="--",
              label=f"{prefix}b in a ({len(rows['nn_b'])})")
    left.set_title("Descriptor distance of geometrically matched keypoints",
                   fontsize=10)
    left.set_xlabel("descriptor distance")
    right.set_title("Position distance to the most similar descriptor",
                    fontsize=10)
    right.set_xlabel("position distance (px), below 0.01 drawn at 0.01")
    log_axis(right, "x", 0.01, 1.2 * max(
        [float(r["pos_dist"]) for _, rows in series
         for r in rows["nn_a"] + rows["nn_b"]], default=1.0))
    right.axvline(match_radius, color="grey", linewidth=0.8)
    for ax in (left, right):
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("share of keypoints")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png
