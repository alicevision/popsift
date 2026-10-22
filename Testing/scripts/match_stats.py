"""Statistics and reports from the match files of popsift-compareSiftFiles.

For every image, popsift-compareSiftFiles --match-dir writes matched.csv,
only_a.csv and only_b.csv, where a is the run under test and b the reference,
and desc_nn_a.csv / desc_nn_b.csv, the descriptor nearest neighbour of every
feature of a in b and of b in a. This module turns them into statistics per
image and for a whole image set, and writes summary.csv, summary.json and
report.md. It decides nothing: pass/fail limits are applied by the caller
(see check_limits).

    recall    = matched / features of b   (share of the reference found)
    precision = matched / features of a   (share of a's features confirmed)
    same pose = matched pairs within SAME_POSE_PX and SAME_POSE_DEG; their
                descriptors should be nearly identical
    descriptor distance = L2 distance of two 128-value descriptors in byte
                units (0-255 per value)
    nearest-descriptor agreement a (b) = share of the matched keypoints of
                a (b) whose most similar descriptor in the other run is their
                matched keypoint; anything below 1 means that the most
                similar descriptor belongs to a keypoint somewhere else
    nearest descriptor in place a (b) = keypoints of a (b) whose most similar
                descriptor in the other run belongs to a keypoint within the
                matching radius max(1 px, 0.5 sigma); "distinctive" ones also
                pass Lowe's ratio test, most similar < 0.8 * second most
                similar descriptor distance

Metric names carry their unit: _px pixels, _deg degrees, _l2 descriptor
distance.

Standard library only, Python 3.8 or newer.
"""

from __future__ import annotations

import csv
import json
import math
import os

# A matched pair has the same pose if it is at most this far apart.
SAME_POSE_PX = 0.25
SAME_POSE_DEG = 2.0

# The matching radius, max(MATCH_RADIUS_MIN_PX, MATCH_RADIUS_SIGMA * sigma),
# as popsift-compareSiftFiles uses it by default, and Lowe's ratio.
MATCH_RADIUS_MIN_PX = 1.0
MATCH_RADIUS_SIGMA = 0.5
LOWE_RATIO = 0.8

# The files of a match folder, by source name.
SOURCES = {
    "matched": "matched.csv",
    "nn_a": "desc_nn_a.csv",
    "nn_b": "desc_nn_b.csv",
}

_DESC_EDGES = [0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, math.inf]
_NN_POS_EDGES = [0, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 500, math.inf]

# Quantities as named in summary.json and summary.csv, with their unit in the
# name (px: pixels, deg: degrees, l2: L2 distance of descriptors in byte
# units): the source rows ("same_pose" is a subset of "matched"), the column,
# a label for readers and the histogram bin edges.
METRICS = {
    "pos_dist_px":  ("matched", "pos_dist", "pixel distance (px)",
                     [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, math.inf]),
    "dx_px":        ("matched", "dx", "offset x, b - a (px)",
                     [-math.inf, -1, -0.5, -0.1, -0.01, 0.01, 0.1, 0.5, 1, math.inf]),
    "dy_px":        ("matched", "dy", "offset y, b - a (px)",
                     [-math.inf, -1, -0.5, -0.1, -0.01, 0.01, 0.1, 0.5, 1, math.inf]),
    "scale_ratio":  ("matched", "scale_ratio", "scale ratio (b/a)",
                     [0, 0.8, 0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.25, math.inf]),
    "ori_diff_deg": ("matched", "ori_diff_deg", "rotation difference, b - a (deg)",
                     [-180, -45, -10, -5, -1, -0.1, 0.1, 1, 5, 10, 45, 180]),
    "abs_ori_diff_deg":
                    ("matched", "ori_diff_deg", "|rotation difference| (deg)",
                     [0, 0.1, 0.5, 1, 2, 5, 10, 20, 45, 90, 180]),
    "desc_dist_l2": ("matched", "desc_dist", "descriptor distance (L2)",
                     _DESC_EDGES),
    "same_pose_desc_dist_l2":
                    ("same_pose", "desc_dist",
                     "descriptor distance (L2), same pose", _DESC_EDGES),
    "nn_a_pos_dist_px":
                    ("nn_a", "pos_dist",
                     "a to the most similar descriptor in b: pixel distance (px)",
                     _NN_POS_EDGES),
    "nn_b_pos_dist_px":
                    ("nn_b", "pos_dist",
                     "b to the most similar descriptor in a: pixel distance (px)",
                     _NN_POS_EDGES),
    "nn_a_desc_dist_l2":
                    ("nn_a", "desc_dist",
                     "a to the most similar descriptor in b: descriptor distance (L2)",
                     _DESC_EDGES),
    "nn_b_desc_dist_l2":
                    ("nn_b", "desc_dist",
                     "b to the most similar descriptor in a: descriptor distance (L2)",
                     _DESC_EDGES),
}

QUANTILES = (("median", 0.5), ("p90", 0.9), ("p99", 0.99))


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_rows(match_dir):
    """The rows of a match folder by source, including "same_pose".

    Missing descriptor nearest neighbour files give empty lists.
    """
    rows = {}
    for source, name in SOURCES.items():
        path = os.path.join(match_dir, name)
        rows[source] = read_csv(path) if source == "matched" or \
            os.path.exists(path) else []
    rows["same_pose"] = [
        r for r in rows["matched"]
        if float(r["pos_dist"]) <= SAME_POSE_PX
        and abs(float(r["ori_diff_deg"])) <= SAME_POSE_DEG]
    return rows


def metric_values(rows, key):
    """The values of a metric from read_rows() output."""
    source, column = METRICS[key][0], METRICS[key][1]
    values = [float(r[column]) for r in rows[source]]
    if key == "abs_ori_diff_deg":
        values = [abs(x) for x in values]
    return values


def _nn_agreement(nn_rows):
    """Share of the queries with a matched keypoint whose descriptor nearest
    neighbour is that match, and the number of such queries."""
    with_geo = [r for r in nn_rows if int(r["geo_index"]) >= 0]
    if not with_geo:
        return None, 0
    agree = sum(1 for r in with_geo if r["n_index"] == r["geo_index"])
    return agree / len(with_geo), len(with_geo)


def _nn_in_place(nn_rows):
    """The number of queries whose descriptor nearest neighbour is within the
    matching radius, and how many of them also pass Lowe's ratio test."""
    in_place = [r for r in nn_rows
                if float(r["pos_dist"]) <= max(
                    MATCH_RADIUS_MIN_PX,
                    MATCH_RADIUS_SIGMA * float(r["q_sigma"]))]
    distinctive = sum(1 for r in in_place
                      if float(r["desc_dist"])
                      < LOWE_RATIO * float(r["second_desc_dist"]))
    return len(in_place), distinctive


def _merge_rows(per_image):
    merged = {}
    for s in per_image:
        for source, rows in s["_rows"].items():
            merged.setdefault(source, []).extend(rows)
    return merged


def _quantile(sorted_values, q):
    """Linear interpolation between closest ranks, like numpy's default."""
    if not sorted_values:
        return None
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def describe(values, edges):
    """Summary statistics and a histogram of a list of numbers."""
    v = sorted(values)
    out = {"n": len(v)}
    if v:
        out["mean"] = sum(v) / len(v)
        for name, q in QUANTILES:
            out[name] = _quantile(v, q)
        out["min"] = v[0]
        out["max"] = v[-1]
    else:
        out.update({k: None for k in ("mean", "median", "p90", "p99",
                                      "min", "max")})
    counts = [0] * (len(edges) - 1)
    for x in v:
        for i in range(len(counts)):
            last = i == len(counts) - 1
            if edges[i] <= x < edges[i + 1] or (last and x == edges[i + 1]):
                counts[i] += 1
                break
    out["histogram"] = {"edges": [_json_num(e) for e in edges],
                        "counts": counts}
    return out


def _json_num(x):
    if x == math.inf:
        return "inf"
    if x == -math.inf:
        return "-inf"
    return x


def image_stats(match_dir):
    """Statistics of one image from the files in match_dir."""
    rows = read_rows(match_dir)
    only_a = read_csv(os.path.join(match_dir, "only_a.csv"))
    only_b = read_csv(os.path.join(match_dir, "only_b.csv"))
    return _stats(rows, len(rows["matched"]) + len(only_a),
                  len(rows["matched"]) + len(only_b))


def _stats(rows, n_a, n_b):
    n_m = len(rows["matched"])
    agree_a, _ = _nn_agreement(rows["nn_a"])
    agree_b, _ = _nn_agreement(rows["nn_b"])
    in_place_a, distinctive_a = _nn_in_place(rows["nn_a"])
    in_place_b, distinctive_b = _nn_in_place(rows["nn_b"])
    return {
        "counts": {"a": n_a, "b": n_b, "matched": n_m,
                   "only_a": n_a - n_m, "only_b": n_b - n_m,
                   "same_pose": len(rows["same_pose"]),
                   "nn_in_place_a": in_place_a,
                   "nn_distinctive_a": distinctive_a,
                   "nn_in_place_b": in_place_b,
                   "nn_distinctive_b": distinctive_b},
        "recall": n_m / n_b if n_b else None,
        "precision": n_m / n_a if n_a else None,
        "nn_agreement_a": agree_a,
        "nn_agreement_b": agree_b,
        "metrics": {k: describe(metric_values(rows, k), METRICS[k][3])
                    for k in METRICS},
        "_rows": rows,
    }


def set_stats(per_image):
    """Statistics of a set: pooled over all pairs, and spread over images.

    per_image is a list of image_stats() results.
    """
    n_a = sum(s["counts"]["a"] for s in per_image)
    n_b = sum(s["counts"]["b"] for s in per_image)
    total = _stats(_merge_rows(per_image), n_a, n_b)

    def spread(getter):
        values = sorted(x for x in (getter(s) for s in per_image)
                        if x is not None)
        if not values:
            return {"min": None, "median": None, "max": None}
        return {"min": values[0], "median": _quantile(values, 0.5),
                "max": values[-1]}

    total["images"] = len(per_image)
    total["per_image_spread"] = {
        "recall": spread(lambda s: s["recall"]),
        "precision": spread(lambda s: s["precision"]),
        "nn_agreement_a": spread(lambda s: s["nn_agreement_a"]),
        "nn_agreement_b": spread(lambda s: s["nn_agreement_b"]),
        **{f"{k}_median": spread(lambda s, k=k: s["metrics"][k]["median"])
           for k in ("pos_dist_px", "abs_ori_diff_deg", "desc_dist_l2",
                     "same_pose_desc_dist_l2")},
    }
    return total


def check_limits(stats, min_recall=None, min_precision=None,
                 max_desc_dist=None, min_nn_agreement=None):
    """Returns the list of limits an image (or set) violates.

    None or a negative value disables a limit. The descriptor limit applies
    to the median descriptor distance of the matched pairs, the NN agreement
    limit to both directions.
    """
    problems = []

    def on(limit):
        return limit is not None and limit >= 0

    if on(min_recall) and stats["recall"] is not None \
            and stats["recall"] < min_recall:
        problems.append(f"recall {stats['recall']:.3f} < {min_recall}")
    if on(min_precision) and stats["precision"] is not None \
            and stats["precision"] < min_precision:
        problems.append(f"precision {stats['precision']:.3f} < {min_precision}")
    median = stats["metrics"]["desc_dist_l2"]["median"]
    if on(max_desc_dist) and median is not None and median > max_desc_dist:
        problems.append(f"median descriptor distance (L2) {median:.3f} > "
                        f"{max_desc_dist}")
    for side in ("a", "b"):
        agree = stats[f"nn_agreement_{side}"]
        if on(min_nn_agreement) and agree is not None \
                and agree < min_nn_agreement:
            problems.append(f"nearest-descriptor agreement {side} {agree:.3f} < "
                            f"{min_nn_agreement}")
    return problems


# --- formatting ---------------------------------------------------------------

def _fmt(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def one_line(stats):
    """A short console summary of image or set statistics."""
    c = stats["counts"]
    m = stats["metrics"]
    return (f"{c['a']} vs {c['b']} keypoints, {c['matched']} matched "
            f"(recall {_fmt(stats['recall'])}, precision "
            f"{_fmt(stats['precision'])}), median pixel distance "
            f"{_fmt(m['pos_dist_px']['median'])} px, median |rotation| "
            f"{_fmt(m['abs_ori_diff_deg']['median'], 2)} deg, median "
            f"descriptor distance (L2) {_fmt(m['desc_dist_l2']['median'], 2)} "
            f"(same pose {_fmt(m['same_pose_desc_dist_l2']['median'], 2)}), "
            f"nearest-descriptor agreement "
            f"{_fmt(stats['nn_agreement_a'])} / "
            f"{_fmt(stats['nn_agreement_b'])}")


_CSV_METRICS = ("pos_dist_px", "abs_ori_diff_deg", "ori_diff_deg",
                "scale_ratio", "desc_dist_l2", "same_pose_desc_dist_l2",
                "nn_a_pos_dist_px", "nn_b_pos_dist_px")

CSV_COLUMNS = (
    ["image", "keypoints_a", "keypoints_b", "matched", "same_pose", "recall",
     "precision", "nn_agreement_a", "nn_agreement_b"]
    + [f"{k}_{q}" for k in _CSV_METRICS
       for q in ("mean", "median", "p90", "max")]
)


def _csv_row(name, stats):
    c = stats["counts"]
    row = [name, c["a"], c["b"], c["matched"], c["same_pose"],
           _fmt(stats["recall"], 4), _fmt(stats["precision"], 4),
           _fmt(stats["nn_agreement_a"], 4), _fmt(stats["nn_agreement_b"], 4)]
    for k in _CSV_METRICS:
        for q in ("mean", "median", "p90", "max"):
            row.append(_fmt(stats["metrics"][k][q], 4))
    return row


def _public(stats):
    return {k: v for k, v in stats.items() if not k.startswith("_")}


def write_report(report_dir, run_a, run_b, set_name, images, total):
    """Writes summary.csv, summary.json and report.md into report_dir.

    images is a list of dicts with 'name', 'input' (image path), 'match_dir'
    and 'stats' (image_stats() result); total is set_stats() of them.
    """
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(report_dir, "summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for im in images:
            w.writerow(_csv_row(im["name"], im["stats"]))
        w.writerow(_csv_row("(set)", total))

    summary = {
        "run_a": run_a,
        "run_b": run_b,
        "set": set_name,
        "definitions": {
            "matched": f"a keypoint of run_a and one of run_b at most "
                       f"max({MATCH_RADIUS_MIN_PX:g} px, "
                       f"{MATCH_RADIUS_SIGMA:g} sigma) apart, with scales "
                       f"within a factor 1.5; paired one to one, closest "
                       f"first",
            "recall": "matched / keypoints of run_b",
            "precision": "matched / keypoints of run_a",
            "pos_dist_px": "distance between the centres of two keypoints "
                           "in pixels of the input image",
            "ori_diff_deg": "orientation of b minus orientation of a, "
                            "wrapped to [-180, 180) degrees",
            "desc_dist_l2": "L2 (Euclidean) distance between two 128-value "
                            "descriptors, in byte units (0-255 per value)",
            "same_pose": f"matched pairs with pixel distance <= "
                         f"{SAME_POSE_PX} px and |ori_diff_deg| <= "
                         f"{SAME_POSE_DEG} deg",
            "nn_agreement_a": "nearest-descriptor agreement: share of the "
                              "matched keypoints of run_a whose most similar "
                              "descriptor in run_b is their matched keypoint",
            "nn_agreement_b": "the same for run_b in run_a",
            "nn_in_place_a": "keypoints of run_a whose most similar "
                             "descriptor in run_b belongs to a keypoint "
                             "within the matching radius",
            "nn_distinctive_a": f"those of nn_in_place_a that also pass "
                                f"Lowe's ratio test: most similar < "
                                f"{LOWE_RATIO:g} * second most similar "
                                f"descriptor distance",
            "nn_in_place_b": "the same for run_b in run_a",
            "nn_distinctive_b": "the same for run_b in run_a",
        },
        "set_stats": _public(total),
        "images": [
            {"name": im["name"],
             "input": os.path.relpath(im["input"], report_dir),
             "match_dir": os.path.relpath(im["match_dir"], report_dir),
             **_public(im["stats"])}
            for im in images
        ],
    }
    with open(os.path.join(report_dir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    with open(os.path.join(report_dir, "report.md"), "w",
              encoding="utf-8") as f:
        f.write(_markdown(report_dir, run_a, run_b, set_name, images, total))


def _markdown(report_dir, run_a, run_b, set_name, images, total):
    out = [f"# {run_a} vs {run_b} on {set_name}", ""]
    out.append(f"a = `{run_a}`, b = `{run_b}`. Recall = matched / keypoints "
               "of b, precision = matched / keypoints of a. Rotation "
               "difference is b - a. Descriptor distance is the L2 distance "
               "of the 128 descriptor values in byte units (0-255). Same "
               f"pose = matched pairs at most {SAME_POSE_PX} px and "
               f"{SAME_POSE_DEG} deg apart. Nearest-descriptor agreement a = "
               "share of the matched keypoints of a whose most similar "
               "descriptor in b is their matched keypoint (b likewise).")
    out.append("")
    out.append("## Set")
    out.append("")
    out.append(f"{total['images']} images. " + one_line(total) + ".")
    out.append("")
    out.append("| Quantity | mean | median | p90 | p99 | max |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for k, (_, _, label, _) in METRICS.items():
        m = total["metrics"][k]
        label = label.replace("|", "\\|")
        out.append(f"| {label} | {_fmt(m['mean'])} | {_fmt(m['median'])} | "
                   f"{_fmt(m['p90'])} | {_fmt(m['p99'])} | {_fmt(m['max'])} |")
    out.append("")
    sp = total["per_image_spread"]
    out.append("Spread over images (min / median / max):")
    out.append("")
    for k, label in (("recall", "recall"), ("precision", "precision"),
                     ("nn_agreement_a", "nearest-descriptor agreement a"),
                     ("nn_agreement_b", "nearest-descriptor agreement b"),
                     ("pos_dist_px_median", "median pixel distance (px)"),
                     ("abs_ori_diff_deg_median",
                      "median |rotation difference| (deg)"),
                     ("desc_dist_l2_median",
                      "median descriptor distance (L2)"),
                     ("same_pose_desc_dist_l2_median",
                      "median descriptor distance (L2), same pose")):
        s = sp[k]
        out.append(f"- {label}: {_fmt(s['min'])} / {_fmt(s['median'])} / "
                   f"{_fmt(s['max'])}")
    out.append("")
    for png, alt in (("cdf.png", "Cumulative distributions"),
                     ("sorted.png", "Sorted differences"),
                     ("descriptor_bins.png", "Descriptor differences per bin")):
        if os.path.exists(os.path.join(report_dir, png)):
            out.append(f"![{alt}]({png})")
            out.append("")
    out.append("## Images")
    out.append("")
    out.append("| Image | a | b | matched | recall | precision | "
               "median pixel dist. (px) | median \\|rot\\| (deg) | "
               "mean rot (deg) | median desc. dist. (L2) | same pose | "
               "median desc. dist. (L2), same pose | "
               "nearest-desc. agreement a | nearest-desc. agreement b |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
               "---: | ---: | ---: | ---: | ---: | ---: |")
    for im in images:
        s = im["stats"]
        c = s["counts"]
        m = s["metrics"]
        out.append(f"| {im['name']} | {c['a']} | {c['b']} | {c['matched']} | "
                   f"{_fmt(s['recall'])} | {_fmt(s['precision'])} | "
                   f"{_fmt(m['pos_dist_px']['median'])} | "
                   f"{_fmt(m['abs_ori_diff_deg']['median'], 2)} | "
                   f"{_fmt(m['ori_diff_deg']['mean'], 2)} | "
                   f"{_fmt(m['desc_dist_l2']['median'], 2)} | "
                   f"{c['same_pose']} | "
                   f"{_fmt(m['same_pose_desc_dist_l2']['median'], 2)} | "
                   f"{_fmt(s['nn_agreement_a'])} | "
                   f"{_fmt(s['nn_agreement_b'])} |")
    out.append("")
    return "\n".join(out)
