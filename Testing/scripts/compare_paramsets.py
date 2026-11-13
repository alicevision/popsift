#!/usr/bin/env python3
"""Compare several PopSift parameter sets with each other, by hand.

This runs every parameter set once per image, then compares every pair of
sets (loop vs grid, loop vs igrid, ...), or every set against one --baseline,
and prints a table of the three agreement levels of the co-located
keypoints (position and scale, also orientation, also descriptor) with
position, rotation and descriptor differences (see match_stats.py). Each pair also
gets a report folder per image set, as with popsift_compare.py.
Meant for developers, not used by CTest.

    compare_paramsets.py --build build --sets loop,grid,igrid,notile \\
                         --images oxford/boat
    compare_paramsets.py --build build --sets popsift-vlfeat,loop \\
                         --baseline vlfeat --images handcrafted

Sets are defined in Testing/paramsets.json; 'vlfeat' runs VLFeat's sift CLI.
A set can carry overrides, e.g. loop@gauss-mode=vlfeat. Images are a
configured image set or a folder.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import match_stats  # noqa: E402
import popsift_compare as pc  # noqa: E402


def _split(text):
    return [s for s in (t.strip() for t in text.split(",")) if s]


def _pairs_and_specs(parser, sets, baseline):
    """The pairs to compare and every set that must run."""
    if baseline:
        sets = [s for s in sets if s != baseline]
        return [(s, baseline) for s in sets], sets + [baseline]
    if len(sets) < 2:
        parser.error("give at least two sets, or a --baseline")
    return list(itertools.combinations(sets, 2)), sets


class Collector:
    """Runs and compares the sets image by image and collects the results."""

    def __init__(self, runner, specs, pairs):
        self.runner = runner
        self.specs = specs
        self.pairs = pairs
        # pair -> list of report entries, one per image
        self.results = {p: [] for p in pairs}
        self.counts = {s: [] for s in specs}
        self.failures = []

    def image_set(self, tools, images, work_dir):
        """Runs and compares every image of one image set or folder."""
        name, folder = pc.resolve_images(tools, images)
        files = pc.find_images(folder)
        if not files:
            raise pc.CompareError(f"no .pgm/.ppm images under {folder}")
        expectations = pc.load_expectations(folder)
        set_dir = os.path.join(work_dir, pc.safe_name(name.replace("/", "_")))
        for index, src in enumerate(files, 1):
            rel = os.path.relpath(src, folder)
            print(f"\n==================== [{index}/{len(files)}] {name}: "
                  f"{rel} ====================", flush=True)
            allow_empty = bool(expectations.get(
                os.path.normpath(rel), {}).get("allow_empty"))
            self.image(src, name, rel, set_dir, allow_empty)

    def image(self, src, name, rel, set_dir, allow_empty):
        """Runs every set on one image and compares the pairs."""
        label = f"{name}: {rel}"
        image_dir = os.path.join(set_dir,
                                 pc.safe_name(os.path.splitext(rel)[0]))
        os.makedirs(image_dir, exist_ok=True)
        pgm = pc.prepare_input(src, image_dir)
        runs = {s: self.runner.run(s, pgm, image_dir) for s in self.specs}
        for s, r in runs.items():
            if r is None:
                self.failures.append(f"{label}: {s} failed")
            elif r is pc.SKIP:
                self.failures.append(f"{label}: {s} skipped, no usable GPU")
            else:
                self.counts[s].append(r.count)
        runs = {s: r for s, r in runs.items() if r is not pc.SKIP}
        for a, b in self.pairs:
            if runs.get(a) is None or runs.get(b) is None:
                continue
            res = self.runner.compare(runs[a], runs[b], image_dir, -1.0,
                                      allow_empty)
            if res.stats is None:
                self.failures.append(f"{label}: comparing {a} with {b} failed")
            else:
                self.results[(a, b)].append(
                    {"name": f"{name}/{rel}", "input": pgm,
                     "pair_dir": res.pair_dir, "stats": res.stats})


def _print_counts(specs, counts):
    print("\nKeypoints per image (mean):")
    for s in specs:
        c = counts[s]
        mean = f"{sum(c) / len(c):.0f}" if c else "-"
        print(f"  {s:24} {mean:>8}   ({len(c)} image(s))")


def _pair_line(pair, label_w, total):
    """One row of the summary table."""
    m = total["metrics"]
    f = match_stats._fmt
    return (f"  {pair:{label_w}}  {f(total['recall']):>7} "
            f"{f(total['precision']):>7} "
            f"{f(total['ori_agreement']):>7} "
            f"{f(total['desc_agreement']):>8} "
            f"{f(m['pos_dist_px']['median']):>8} "
            f"{f(m['abs_ori_diff_deg']['median'], 2):>9} "
            f"{f(m['ori_diff_deg']['mean'], 2):>8} "
            f"{f(m['desc_dist_l2']['median'], 2):>8} "
            f"{f(m['same_ori_desc_dist_l2']['median'], 2):>8} "
            f"{f(total['nn_agreement_a']):>8} "
            f"{f(total['nn_agreement_b']):>8}")


def _summarize(pairs, results, work_dir, images):
    """Prints the table of all pairs and writes their reports and plots."""
    print("\nCo-located keypoints of A vs B, pooled over all images "
          "(recall = co-located/B, precision = co-located/A; rotation "
          "is B - A):")
    label_w = max(len(f"{a} vs {b}") for a, b in pairs)
    print(f"  {'':{label_w}}  {'recall':>7} {'precis.':>7} "
          f"{'ori agr':>7} {'desc agr':>8} "
          f"{'med px':>8} {'med |rot|':>9} {'mean rot':>8} "
          f"{'med desc':>8} {'same ori':>8} {'agree a':>8} "
          f"{'agree b':>8}")
    plotting = pc.load_plotting()
    series = []
    for a, b in pairs:
        entries = results[(a, b)]
        pair = f"{a} vs {b}"
        if not entries:
            print(f"  {pair:{label_w}}  (no results)")
            continue
        total = match_stats.set_stats([e["stats"] for e in entries])
        print(_pair_line(pair, label_w, total))
        report_dir = os.path.join(
            work_dir, f"report_{pc.safe_name(a)}__vs__{pc.safe_name(b)}")
        if plotting is not None:
            os.makedirs(report_dir, exist_ok=True)
            plotting.render_agreement(
                os.path.join(report_dir, "agreement.png"),
                f"{pair} on {images}", total["_rows"], total["counts"],
                **pc.AGREEMENT_TOLERANCES)
            plotting.render_cdfs(os.path.join(report_dir, "cdf.png"),
                                 f"{pair} on {images}",
                                 [(pair, total["_rows"])])
            series.append((pair, total["_rows"]))
        match_stats.write_report(report_dir, a, b, images, entries, total)
    print("  (ori agr: share of co-located pairs with the same "
          f"orientation, at most {match_stats.ORI_TOLERANCE_DEG:g} deg "
          "apart;\n   desc agr: share of those with a similar "
          "descriptor, descriptor distance (L2) at most "
          f"{match_stats.DESC_TOLERANCE_L2:g};\n   med px: median "
          "pixel distance; med desc: median descriptor distance (L2);\n"
          "   same ori: the same for pairs with the same orientation;\n"
          "   agree: nearest-descriptor agreement, the share of "
          "co-located keypoints whose most\n   similar descriptor in the "
          "other run is their co-located keypoint)")
    print(f"\nReports: {work_dir}/report_<A>__vs__<B>/report.md")
    if len(series) > 1:
        out = os.path.join(work_dir, "cdf_all_pairs.png")
        plotting.render_cdfs(out, f"All pairs on {images}", series)
        print(f"All pairs in one plot: {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    parser.add_argument("--build", required=True,
                        help="the CMake build directory")
    parser.add_argument("--sets",
                        help="comma-separated parameter sets to compare")
    parser.add_argument("--baseline",
                        help="compare every set against this one instead of "
                             "comparing all pairs, e.g. vlfeat")
    parser.add_argument("--images", default="handcrafted",
                        help="comma-separated image sets or folders "
                             "(default: %(default)s)")
    parser.add_argument("--work-dir",
                        help="where outputs go (default: "
                             "<build>/Testing/paramset-runs)")
    parser.add_argument("--list", action="store_true",
                        help="list the parameter sets and image sets, and exit")
    args = parser.parse_args(argv)

    try:
        tools = pc.load_tools(pc.find_tools_file(args.build))
        if args.list:
            pc.print_list(tools)
            return 0
        if not args.sets:
            parser.error("--sets is required (or use --list)")
        pairs, specs = _pairs_and_specs(parser, _split(args.sets),
                                        args.baseline)
        runner = pc.Runner(tools)
        runner.check(specs)
        work_dir = args.work_dir or os.path.join(
            os.path.abspath(args.build), "Testing", "paramset-runs")

        collector = Collector(runner, specs, pairs)
        for images in _split(args.images):
            collector.image_set(tools, images, work_dir)

        print("\n==================== Summary ====================")
        print(f"Outputs: {work_dir}")
        _print_counts(specs, collector.counts)
        _summarize(pairs, collector.results, work_dir, args.images)
        if collector.failures:
            print("\nProblems:")
            for f in collector.failures:
                print("  " + f)
            return 1
        return 0
    except pc.CompareError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
