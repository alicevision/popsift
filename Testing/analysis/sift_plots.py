"""Drawing SIFT keypoints over their input image, with numpy and matplotlib.

Used by Testing/scripts/popsift_compare.py and compare_paramsets.py
(agreement.png, unpaired.png, cdf.png, sorted.png, descriptor_bins.png and
image_heatmap.png) and by inspect_matches.py. Keypoints are drawn with the
classic SIFT glyph: a
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, FuncFormatter

# Colour-blind safe pair (Okabe-Ito): run a in orange, run b in blue.
COLOR_A = "#E69F00"
COLOR_B = "#0072B2"

# Colour-blind safe colours (Okabe-Ito) for several curves in one plot.
SERIES_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
                 "#D55E00", "#F0E442", "#000000"]

# The agreement levels of co-located pairs as in colocated.csv, lowest first,
# with a label and a colour (Okabe-Ito): vermillion where the orientation
# differs, reddish purple where only the descriptor differs, green where all
# agree.
LEVELS = ("position", "orientation", "descriptor")
LEVEL_LABELS = {"position": "orientation differs",
                "orientation": "only the descriptor differs",
                "descriptor": "orientation and descriptor agree"}
LEVEL_COLORS = {"position": "#D55E00", "orientation": "#CC79A7",
                "descriptor": "#009E73"}

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


# The columns of colocated.csv that the explorer and the pair inspector use,
# and "level", the agreement column as an index into LEVELS.
PAIR_COLUMNS = ("a_x", "a_y", "a_sigma", "a_ori_deg",
                "b_x", "b_y", "b_sigma", "b_ori_deg",
                "pos_dist", "scale_ratio", "ori_diff_deg", "desc_dist")
PAIR_INDEX_COLUMNS = ("a_index", "b_index")
PAIRS_NPZ = "pairs.npz"


def pairs_from_csv(path):
    """The PAIR_COLUMNS, PAIR_INDEX_COLUMNS and level of a colocated.csv as
    numpy arrays (float32 and int32), keyed by column name."""
    rows = read_rows(path)
    out = {c: np.array([float(r[c]) for r in rows], dtype=np.float32)
           for c in PAIR_COLUMNS}
    out.update({c: np.array([int(r[c]) for r in rows], dtype=np.int32)
                for c in PAIR_INDEX_COLUMNS})
    out["level"] = np.array([LEVELS.index(r["agreement"]) for r in rows],
                            dtype=np.int32)
    return out


def read_pairs(pair_dir):
    """The co-located pairs of a pair folder: from pairs.npz, which the
    explorer package holds instead of the CSV files, or from colocated.csv."""
    npz = os.path.join(pair_dir, PAIRS_NPZ)
    if os.path.exists(npz):
        with np.load(npz) as data:
            return {k: data[k] for k in data.files}
    return pairs_from_csv(os.path.join(pair_dir, "colocated.csv"))


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


def render_unpaired(pgm, pair_dir, out_png, label_a="a", label_b="b"):
    """Writes an image of the keypoints that only one of the runs found, that
    is, without a co-located keypoint in the other run."""
    image = read_pgm(pgm)
    only_a = read_rows(os.path.join(pair_dir, "only_a.csv"))
    only_b = read_rows(os.path.join(pair_dir, "only_b.csv"))
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
    axis.set_major_formatter(FuncFormatter(
        lambda v, _: np.format_float_positional(v, trim="-")))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.grid(True, which="major", axis=which, alpha=0.5)
    ax.grid(True, which="minor", axis=which, alpha=0.2)


def render_agreement(out_png, title, rows, counts, ori_tolerance=2.0,
                     desc_tolerance=12.0):
    """Writes the three agreement levels of the co-located pairs.

    Left: every co-located pair by its rotation difference and descriptor
    distance (L2), coloured by agreement level. The grey lines are the
    tolerances, and the three regions are what a tester can conclude: right
    of the orientation tolerance the orientation computation differs and the
    descriptor cannot be judged; left of it and above the descriptor
    tolerance the descriptor extraction is suspect; below it all agree.
    Right: the keypoints of a and b split into unpaired, and co-located by
    agreement level. rows is match_stats.read_rows() output and counts the
    counts of match_stats.image_stats() or set_stats(), possibly merged over
    images. Values outside an axis are drawn at its ends.
    """
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [3, 2]})
    pairs = rows["colocated"]
    rot = np.array([abs(float(r["ori_diff_deg"])) for r in pairs])
    desc = np.array([float(r["desc_dist"]) for r in pairs])
    level = np.array([r["agreement"] for r in pairs])
    rot_lo, desc_lo = 0.01, 0.1
    desc_hi = 1.2 * max([10 * desc_tolerance, *desc])
    for lv in LEVELS:
        sel = level == lv
        left.plot(np.clip(rot[sel], rot_lo, 180.0),
                  np.clip(desc[sel], desc_lo, desc_hi), ".",
                  markersize=3, alpha=0.6, color=LEVEL_COLORS[lv],
                  label=f"{LEVEL_LABELS[lv]} ({int(sel.sum())})")
    log_axis(left, "x", rot_lo, 180.0)
    log_axis(left, "y", desc_lo, desc_hi)
    left.axvline(ori_tolerance, color="grey", linewidth=0.8)
    left.plot([rot_lo, ori_tolerance], [desc_tolerance, desc_tolerance],
              color="grey", linewidth=0.8)
    box = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8,
           "edgecolor": "none"}
    left.text(ori_tolerance * 1.15, desc_hi / 1.3,
              "orientation differs:\ndescriptor cannot be judged",
              fontsize=8, va="top", bbox=box)
    left.text(rot_lo * 1.15, desc_hi / 1.3,
              "same orientation,\ndescriptor differs:\nsuspect descriptor",
              fontsize=8, va="top", bbox=box)
    left.text(rot_lo * 1.15, desc_lo * 1.3, "all agree", fontsize=8,
              va="bottom", bbox=box)
    left.set_xlabel(f"|rotation difference| (deg), below {rot_lo:g} at "
                    f"{rot_lo:g}; tolerance {ori_tolerance:g}")
    left.set_ylabel(f"descriptor distance (L2), below {desc_lo:g} at "
                    f"{desc_lo:g}; tolerance {desc_tolerance:g}")
    left.set_title("Co-located pairs: rotation and descriptor difference",
                   fontsize=10)
    left.legend(fontsize=8, loc="lower right", framealpha=1)

    c = counts
    segments = [("unpaired", "#BBBBBB", c["only_a"], c["only_b"])] + \
        [(LEVEL_LABELS[lv], LEVEL_COLORS[lv], n, n) for lv, n in
         (("position", c["orientation_differs"]),
          ("orientation", c["descriptor_differs"]),
          ("descriptor", c["similar_descriptor"]))]
    left_edge = np.zeros(2)
    # Numbers only on segments wide enough for them.
    min_labelled = 0.06 * max(c["a"], c["b"], 1)
    for label, color, n_a, n_b in segments:
        widths = np.array([n_a, n_b], dtype=float)
        right.barh([1, 0], widths, left=left_edge, color=color, label=label)
        for y, w, x in zip((1, 0), widths, left_edge):
            if w >= min_labelled:
                right.text(x + w / 2, y, str(int(w)), ha="center",
                           va="center", fontsize=8)
        left_edge += widths
    right.set_yticks([1, 0], [f"a ({c['a']})", f"b ({c['b']})"])
    right.set_xlabel("keypoints")
    right.set_title("Keypoints by agreement level", fontsize=10)
    right.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15),
                 ncol=2, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png


def render_cdfs(out_png, title, series, pair_radius=1.0):
    """Writes cumulative distributions of descriptor and pixel distances.

    series is a list of (label, rows) where rows is match_stats.read_rows()
    output, possibly merged over images. Left: descriptor distance of the
    co-located pairs, solid for the pairs with the same orientation, dashed
    for all. Right: pixel distance from each keypoint to the keypoint
    with the most similar descriptor in the other run, solid a in b, dashed
    b in a, on a logarithmic axis from 0.01 px (smaller distances are drawn
    at 0.01 px); the grey line is the smallest pairing radius.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, (label, rows) in enumerate(series):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        prefix = f"{label}: " if len(series) > 1 else ""
        same = rows["same_orientation"]
        _ecdf(left, [float(r["desc_dist"]) for r in same],
              color=color, label=f"{prefix}same orientation ({len(same)})")
        _ecdf(left, [float(r["desc_dist"]) for r in rows["colocated"]],
              color=color, linestyle="--",
              label=f"{prefix}all co-located ({len(rows['colocated'])})")
        _ecdf(right, [max(float(r["pos_dist"]), 0.01) for r in rows["nn_a"]],
              color=color, label=f"{prefix}a in b ({len(rows['nn_a'])})")
        _ecdf(right, [max(float(r["pos_dist"]), 0.01) for r in rows["nn_b"]],
              color=color, linestyle="--",
              label=f"{prefix}b in a ({len(rows['nn_b'])})")
    left.set_title("Descriptor distance (L2) of co-located keypoints",
                   fontsize=10)
    left.set_xlabel("descriptor distance (L2)")
    right.set_title("Pixel distance to the most similar descriptor",
                    fontsize=10)
    right.set_xlabel("pixel distance (px), below 0.01 drawn at 0.01")
    log_axis(right, "x", 0.01, 1.2 * max(
        [float(r["pos_dist"]) for _, rows in series
         for r in rows["nn_a"] + rows["nn_b"]], default=1.0))
    right.axvline(pair_radius, color="grey", linewidth=0.8)
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


# The plots of differences from the 2021 comparison scripts, redone on the
# pair files: sorted.png, descriptor_bins.png and image_heatmap.png.

DESC_BINS_CSV = "desc_bins.csv"

# Diverging colour map for b - a: orange where a is larger, a neutral grey at
# 0, blue where b is larger, like the glyph colours of the two runs.
DIVERGING = LinearSegmentedColormap.from_list(
    "a_grey_b", [COLOR_A, "#e8e8e8", COLOR_B])


def read_desc_bins(pair_dir):
    """desc_bins.csv of popsift-compareSiftFiles as {subset: (pairs,
    sum_diff, sum_abs_diff)} with arrays of 128 values, or None without it."""
    path = os.path.join(pair_dir, DESC_BINS_CSV)
    if not os.path.exists(path):
        return None
    out = {}
    for r in read_rows(path):
        _, s, a = out.setdefault(
            r["subset"], [0, np.zeros(128), np.zeros(128)])
        i = int(r["element"])
        out[r["subset"]][0] = int(r["pairs"])
        s[i] = float(r["sum_diff"])
        a[i] = float(r["sum_abs_diff"])
    return {k: tuple(v) for k, v in out.items()}


def merge_desc_bins(bins_list):
    """The desc_bins of several images, summed; None if none has any."""
    bins_list = [b for b in bins_list if b]
    if not bins_list:
        return None
    return {k: (sum(b[k][0] for b in bins_list),
                sum(b[k][1] for b in bins_list),
                sum(b[k][2] for b in bins_list))
            for k in bins_list[0]}


def _sorted_panel(ax, values, bottom, top, ylabel, title, **kwargs):
    v = np.sort(np.asarray(values, dtype=float))
    ax.plot(np.arange(1, len(v) + 1), np.clip(v, bottom, top), **kwargs)
    log_axis(ax, "y", bottom, top)
    ax.set_xlim(0, max(len(v), 1))
    ax.set_xlabel("keypoints, sorted by the value")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    return v


def render_sorted(out_png, title, rows):
    """Writes the three sorted plots of 2021: for every keypoint of a, the
    descriptor distance to the most similar descriptor of b with the second
    most similar one in the same order; the pixel distances and the rotation
    differences of the co-located pairs. rows is match_stats.read_rows()
    output, possibly merged over images. Values outside an axis are drawn at
    its ends."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    nn = rows["nn_a"]
    best = np.array([float(r["desc_dist"]) for r in nn])
    second = np.array([float(r["second_desc_dist"]) for r in nn])
    order = np.argsort(best, kind="stable")
    # 1.0 also stands for empty arrays.
    top = 1.2 * max([1.0, *second[np.isfinite(second)], *best])
    ax = axes[0]
    x = np.arange(1, len(nn) + 1)
    ax.plot(x, np.clip(second[order], 1.0, top), ".", markersize=1.5,
            color=COLOR_A, label="second most similar")
    ax.plot(x, np.clip(best[order], 1.0, top), color=COLOR_B, linewidth=2,
            label="most similar")
    log_axis(ax, "y", 1.0, top)
    ax.set_xlim(0, max(len(nn), 1))
    ax.set_xlabel("keypoints of a, sorted by the most similar")
    ax.set_ylabel("descriptor distance (L2), below 1 at 1")
    ax.set_title("Most similar descriptors in b", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", framealpha=1)
    pairs = rows["colocated"]
    # Co-located keypoints are mostly at the same position up to float
    # rounding, hence the much lower end than elsewhere.
    _sorted_panel(axes[1], [float(r["pos_dist"]) for r in pairs], 1e-5,
                  1.2 * max([float(r["pos_dist"]) for r in pairs],
                            default=1.0),
                  "pixel distance (px), below 0.00001 at 0.00001",
                  "Pixel distance of the co-located pairs", color="#333333",
                  linewidth=2)
    _sorted_panel(axes[2], [abs(float(r["ori_diff_deg"])) for r in pairs],
                  0.01, 180.0, "rotation difference (deg), below 0.01 at 0.01",
                  "Rotation difference of the co-located pairs",
                  color="#333333", linewidth=2)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png


def _bin_grids(values):
    """The 128 descriptor values as 8 grids of 4 x 4 cells, one per
    orientation bin, in VLFeat's order: element = 32 * row + 8 * column +
    bin."""
    return np.asarray(values, dtype=float).reshape(4, 4, 8).transpose(2, 0, 1)


def render_desc_bins(out_png, title, bins):
    """Writes the 2021 descriptor heat maps: per orientation bin, the 4 x 4
    cells of the descriptor with the mean difference b - a of co-located
    pairs. Rows: mean absolute and mean signed difference of the pairs with
    the same orientation, and mean absolute difference of all co-located
    pairs."""
    rows = [("same_orientation", "abs", "same orientation, mean |b - a|"),
            ("same_orientation", "signed", "same orientation, mean b - a"),
            ("colocated", "abs", "all co-located, mean |b - a|")]
    fig, axes = plt.subplots(len(rows), 8, figsize=(15, 6.2))
    for k, (subset, kind, label) in enumerate(rows):
        pairs, s, a = bins.get(subset, (0, np.zeros(128), np.zeros(128)))
        mean = (s if kind == "signed" else a) / max(pairs, 1)
        grids = _bin_grids(mean)
        if kind == "signed":
            lim = max(float(np.abs(grids).max()), 1e-6)
            cmap, vmin, vmax = DIVERGING, -lim, lim
        else:
            cmap, vmin, vmax = "Blues", 0.0, max(float(grids.max()), 1e-6)
        for b in range(8):
            ax = axes[k, b]
            im = ax.imshow(grids[b], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_title(f"bin {b} ({b * 45}°)", fontsize=9)
            if b == 0:
                ax.set_ylabel(f"{label}\n{pairs} pairs", fontsize=8)
        fig.colorbar(im, ax=list(axes[k]), fraction=0.015, pad=0.01)
    fig.suptitle(f"{title}\nDescriptor differences per orientation bin over "
                 "the 4 x 4 cells (byte units)", fontsize=11)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png


def heatmap_cell(width, height):
    """Cell size in pixels for about 32 cells along the long side."""
    return max(4, int(round(max(width, height) / 32)))


def _cell_stat(idx, values, ncells, stat):
    grid = np.full(ncells, np.nan)
    order = np.argsort(idx, kind="stable")
    cells, starts = np.unique(idx[order], return_index=True)
    for c, part in zip(cells, np.split(np.asarray(values)[order], starts[1:])):
        grid[c] = stat(part)
    return grid


def _log_colorbar(fig, im, ax, vmin, vmax):
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                       orientation="horizontal")
    ticks = [10.0 ** k for k in range(math.floor(math.log10(vmin)),
                                      math.ceil(math.log10(vmax)) + 1)
             if vmin <= 10.0 ** k <= vmax]
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"{t:g}" for t in ticks])


def render_image_heatmap(pgm, pairs, out_png, title=""):
    """Writes heat maps over the input image, by the position of a's
    keypoint of every co-located pair: the share of pairs whose orientation
    differs (agreement level "position"), and the 90th percentile of the
    rotation difference and of the descriptor distance, on logarithmic colour
    scales. Cells without pairs stay empty. pairs is read_pairs()
    output."""
    image = read_pgm(pgm)
    h, w = image.shape
    cell = heatmap_cell(w, h)
    nx, ny = -(-w // cell), -(-h // cell)
    cx = np.clip((pairs["a_x"] // cell).astype(int), 0, nx - 1)
    cy = np.clip((pairs["a_y"] // cell).astype(int), 0, ny - 1)
    idx = cy * nx + cx
    rot = np.abs(pairs["ori_diff_deg"])
    desc = pairs["desc_dist"]

    def p90(v):
        return np.quantile(v, 0.9)

    panels = [
        (_cell_stat(idx, pairs["level"] == 0, nx * ny, np.mean), None,
         0.0, 1.0, "share of pairs whose orientation differs"),
        (_cell_stat(idx, rot, nx * ny, p90), "log", 0.01, 180.0,
         "rotation difference (deg), 90th percentile"),
        (_cell_stat(idx, desc, nx * ny, p90), "log", 1.0,
         max([10.0, *desc]),
         "descriptor distance (L2), 90th percentile")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6 * h / w + 1.6),
                             layout="constrained")
    for ax, (grid, scale, vmin, vmax, label) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, alpha=0.5,
                  extent=(0, w, h, 0))
        norm = LogNorm(vmin, vmax) if scale == "log" else None
        im = ax.imshow(np.clip(grid.reshape(ny, nx), vmin, vmax),
                       cmap="Blues", norm=norm,
                       vmin=None if norm else vmin, vmax=None if norm else vmax,
                       extent=(0, nx * cell, ny * cell, 0), alpha=0.85,
                       interpolation="nearest")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=10)
        if norm:
            _log_colorbar(fig, im, ax, vmin, vmax)
        else:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                         orientation="horizontal")
    fig.suptitle(f"{title}\nco-located pairs in cells of {cell} x {cell} px",
                 fontsize=11)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png
