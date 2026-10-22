#!/usr/bin/env python3
"""Compare several PopSift parameter sets with each other, by hand.

This runs every parameter set once per image, then compares every pair of
sets (loop vs grid, loop vs igrid, ...), or every set against one --baseline,
and prints a table of recall, precision, position, rotation and descriptor
differences of the matched features (see match_stats.py). Each pair also
gets a report folder per image set, as with popsift_compare.py.
Meant for developers, not used by CTest.

    compare_paramsets.py --build build --sets loop,grid,igrid,notile \\
                         --images oxford/boat
    compare_paramsets.py --build build --sets vlfeat-emulation,loop \\
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    parser.add_argument("--build", required=True,
                        help="the CMake build directory")
    parser.add_argument("--config",
                        help="the configuration of a multi-config build, "
                             "e.g. Release")
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
        tools = pc.load_tools(pc.find_tools_file(args.build, args.config))
        paramsets = pc.load_paramsets(tools["paramsets"])
        if args.list:
            print("Parameter sets (Testing/paramsets.json):")
            print(f"  {'vlfeat':18} VLFeat's sift CLI")
            for name, entry in sorted(paramsets.items()):
                if not entry.get("abstract"):
                    print(f"  {name:18} {entry.get('description', '')}")
            print("Image sets:")
            for name, folder in sorted(tools["image_sets"].items()):
                print(f"  {name:18} {folder}")
            return 0
        if not args.sets:
            parser.error("--sets is required (or use --list)")

        sets = _split(args.sets)
        if args.baseline:
            sets = [s for s in sets if s != args.baseline]
            pairs = [(s, args.baseline) for s in sets]
            specs = sets + [args.baseline]
        else:
            if len(sets) < 2:
                parser.error("give at least two sets, or a --baseline")
            pairs = list(itertools.combinations(sets, 2))
            specs = sets
        runner = pc.Runner(tools)
        runner.check(specs)

        work_dir = args.work_dir or os.path.join(
            os.path.abspath(args.build), "Testing", "paramset-runs")
        # pair -> list of report entries, one per image
        results = {p: [] for p in pairs}
        counts = {s: [] for s in specs}
        failures = []

        for images in _split(args.images):
            name, folder = pc.resolve_images(tools, images)
            files = pc.find_images(folder)
            if not files:
                raise pc.CompareError(f"no .pgm/.ppm images under {folder}")
            expectations = pc.load_expectations(folder)
            set_dir = os.path.join(work_dir, pc.safe_name(name.replace("/", "_")))
            for index, src in enumerate(files, 1):
                rel = os.path.relpath(src, folder)
                label = f"{name}: {rel}"
                print(f"\n==================== [{index}/{len(files)}] {label} "
                      "====================", flush=True)
                stem = os.path.splitext(rel)[0]
                image_dir = os.path.join(set_dir,
                                         f"{index:04d}-{pc.safe_name(stem)}")
                os.makedirs(image_dir, exist_ok=True)
                pgm = pc.prepare_input(src, image_dir)
                runs = {s: runner.run(s, pgm, image_dir) for s in specs}
                for s, r in runs.items():
                    if r is None:
                        failures.append(f"{label}: {s} failed")
                    else:
                        counts[s].append(r.count)
                for a, b in pairs:
                    if runs[a] is None or runs[b] is None:
                        continue
                    allow_empty = bool(expectations.get(
                        os.path.normpath(rel), {}).get("allow_empty"))
                    res = runner.compare(runs[a], runs[b], image_dir, -1.0,
                                         allow_empty)
                    if res.stats is None:
                        failures.append(f"{label}: comparing {a} with {b} failed")
                    else:
                        results[(a, b)].append(
                            {"name": f"{name}/{rel}", "input": pgm,
                             "match_dir": res.match_dir, "stats": res.stats})

        print("\n==================== Summary ====================")
        print(f"Outputs: {work_dir}")
        print("\nFeatures per image (mean):")
        for s in specs:
            c = counts[s]
            mean = f"{sum(c) / len(c):.0f}" if c else "-"
            print(f"  {s:24} {mean:>8}   ({len(c)} image(s))")
        print("\nMatched features of A vs B, pooled over all images "
              "(recall = matched/B, precision = matched/A; rotation is B - A):")
        label_w = max(len(f"{a} vs {b}") for a, b in pairs)
        print(f"  {'':{label_w}}  {'recall':>7} {'precis.':>7} "
              f"{'med pos':>8} {'med |rot|':>9} {'mean rot':>8} "
              f"{'med desc':>8} {'same pose':>9} {'NN agr a':>8} "
              f"{'NN agr b':>8}")
        plotting = pc.load_plotting()
        series = []
        for a, b in pairs:
            entries = results[(a, b)]
            pair = f"{a} vs {b}"
            if not entries:
                print(f"  {pair:{label_w}}  (no results)")
                continue
            total = match_stats.set_stats([e["stats"] for e in entries])
            m = total["metrics"]
            f = match_stats._fmt
            print(f"  {pair:{label_w}}  {f(total['recall']):>7} "
                  f"{f(total['precision']):>7} "
                  f"{f(m['pos_dist']['median']):>8} "
                  f"{f(m['abs_ori_diff']['median'], 2):>9} "
                  f"{f(m['ori_diff']['mean'], 2):>8} "
                  f"{f(m['desc_dist']['median'], 2):>8} "
                  f"{f(m['same_pose_desc_dist']['median'], 2):>9} "
                  f"{f(total['nn_agreement_a']):>8} "
                  f"{f(total['nn_agreement_b']):>8}")
            report_dir = os.path.join(
                work_dir, f"report_{pc.safe_name(a)}__vs__{pc.safe_name(b)}")
            if plotting is not None:
                os.makedirs(report_dir, exist_ok=True)
                plotting.render_cdfs(os.path.join(report_dir, "cdf.png"),
                                     f"{pair} on {args.images}",
                                     [(pair, total["_rows"])])
                series.append((pair, total["_rows"]))
            match_stats.write_report(report_dir, a, b, args.images, entries,
                                     total)
        print("  (med desc, same pose: median descriptor distance of matched "
              "pairs at most\n   "
              f"{match_stats.SAME_POSE_PX} px and {match_stats.SAME_POSE_DEG} "
              "deg apart; NN agr: share of matched features whose most\n"
              "   similar descriptor in the other run is their match)")
        print(f"\nReports: {work_dir}/report_<A>__vs__<B>/report.md")
        if len(series) > 1:
            out = os.path.join(work_dir, "cdf_all_pairs.png")
            plotting.render_cdfs(out, f"All pairs on {args.images}", series)
            print(f"All pairs in one plot: {out}")
        if failures:
            print("\nProblems:")
            for f in failures:
                print("  " + f)
            return 1
        return 0
    except pc.CompareError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
