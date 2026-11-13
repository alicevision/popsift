#!/usr/bin/env python3
"""Packs the results of a comparison run for the explorer, as one zip file.

    make_explorer_package.py <work dir> [-o popsift-explorer.zip]

<work dir> is where popsift_compare.py or compare_paramsets.py wrote their
results, e.g. <build>/Testing/run. Every report_<a>__vs__<b>
folder below it goes into the package with the images it covers:

    popsift-explorer/
        explorer.py, inspect_matches.py, sift_plots.py   the tools
        README.md, glossary.md, requirements.txt
        data/                                            the work dir layout
            <set>/report_<a>__vs__<b>/   summary.json, summary.csv,
                                         report.md and the set's plots
            <set>/<image>/input.pgm      the image the runs read; .pgm
                                         images from outside the work dir
                                         are copied here, and summary.json
                                         points to the copy
            <set>/<image>/<a>__vs__<b>/  the image's plots and pairs.npz

pairs.npz holds the co-located pairs as float32 and int32 arrays (numpy's
format records the byte order), instead of the CSV files. The CSV files and
the feature files of the runs are not packed; they stay in the work dir.

Needs numpy (Testing/analysis/requirements.txt).
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sift_plots as sp  # noqa: E402

TOOLS = ("explorer.py", "inspect_matches.py", "sift_plots.py", "glossary.md",
         "requirements.txt")
REPORT_FILES = ("summary.json", "summary.csv", "report.md", "agreement.png",
                "cdf.png", "sorted.png", "descriptor_bins.png")
IMAGE_FILES = ("agreement.png", "unpaired.png", "cdf.png", "sorted.png",
               "descriptor_bins.png", "image_heatmap.png")

PACKAGE_README = """# PopSift vs VLFeat explorer

Results of comparing PopSift's SIFT keypoints and descriptors with VLFeat's,
on the test images in data/. Start the explorer with

    python3 explorer.py

It needs Python 3.8 or newer with numpy and matplotlib
(pip install -r requirements.txt). glossary.md explains the terms.

The report.md files in data/ can also be read without the explorer.
"""


def find_reports(work_dir):
    return sorted(os.path.dirname(p) for p in glob.glob(
        os.path.join(work_dir, "**", "report_*", "summary.json"),
        recursive=True))


def build(work_dir, out_zip, top="popsift-explorer"):
    reports = find_reports(work_dir)
    if not reports:
        raise SystemExit(f"no report_*/summary.json below {work_dir}")
    written = set()
    images = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        def arc_of(path):
            rel = os.path.relpath(os.path.normpath(path), work_dir)
            if rel.startswith(".."):
                raise SystemExit(f"{path} is outside {work_dir}")
            return f"{top}/data/{rel}".replace(os.sep, "/")

        def add(path, arc=None):
            arc = arc or arc_of(path)
            if arc not in written and os.path.exists(path):
                z.write(path, arc)
                written.add(arc)

        for name in TOOLS:
            z.write(os.path.join(HERE, name), f"{top}/{name}")
        z.writestr(f"{top}/README.md", PACKAGE_README)
        for report in reports:
            for name in REPORT_FILES[1:]:
                add(os.path.join(report, name))
            with open(os.path.join(report, "summary.json"),
                      encoding="utf-8") as f:
                summary = json.load(f)
            for entry in summary["images"]:
                pair_dir = os.path.normpath(
                    os.path.join(report, entry["pair_dir"]))
                source = os.path.normpath(os.path.join(report,
                                                       entry["input"]))
                if os.path.relpath(source, work_dir).startswith(".."):
                    # An image of the source tree or of a download: copy it
                    # next to the image's pair folders.
                    target = os.path.join(os.path.dirname(pair_dir),
                                          "input.pgm")
                    entry["input"] = os.path.relpath(target, report)
                    add(source, arc_of(target))
                else:
                    add(source)
                for name in IMAGE_FILES:
                    add(os.path.join(pair_dir, name))
                arc = f"{top}/data/" + os.path.relpath(
                    os.path.join(pair_dir, sp.PAIRS_NPZ),
                    work_dir).replace(os.sep, "/")
                if arc not in written:
                    buf = io.BytesIO()
                    np.savez_compressed(buf, **sp.pairs_from_csv(
                        os.path.join(pair_dir, "colocated.csv")))
                    z.writestr(arc, buf.getvalue())
                    written.add(arc)
                    images += 1
            z.writestr(arc_of(os.path.join(report, REPORT_FILES[0])),
                       json.dumps(summary, indent=1))
            print(f"  {os.path.relpath(report, work_dir)}: "
                  f"{len(summary['images'])} image(s)", flush=True)
    size = os.path.getsize(out_zip)
    print(f"{out_zip}: {len(reports)} report(s), {images} image "
          f"comparison(s), {size / 1e6:.1f} MB")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    parser.add_argument("work_dir", help="the work dir of a comparison run")
    parser.add_argument("-o", "--output", default="popsift-explorer.zip",
                        help="the zip file to write (default: %(default)s)")
    args = parser.parse_args(argv)
    build(os.path.abspath(args.work_dir), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
