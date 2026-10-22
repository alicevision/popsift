#!/usr/bin/env python3
"""Packs the results of a comparison run for the explorer, as one zip file.

    make_explorer_package.py <work dir> [-o popsift-explorer.zip]

<work dir> is where popsift_compare.py or compare_paramsets.py wrote their
results, e.g. <build>/Testing/compareVlfeat/run. Every report_<a>__vs__<b>
folder below it goes into the package with the images it covers:

    popsift-explorer/
        explorer.py, inspect_matches.py, sift_plots.py   the tools
        README.md, glossary.md, requirements.txt
        data/                                            the work dir layout
            <set>/report_<a>__vs__<b>/   summary.json, summary.csv,
                                         report.md, cdf.png
            <set>/<image>/input.pgm      the image the runs read
            <set>/<image>/<a>__vs__<b>/  unmatched.png, cdf.png and
                                         matches.npz

matches.npz holds the matched pairs as float32 and int32 arrays (numpy's
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
REPORT_FILES = ("summary.json", "summary.csv", "report.md", "cdf.png")
IMAGE_FILES = ("unmatched.png", "cdf.png")

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
        def add(path):
            rel = os.path.relpath(os.path.normpath(path), work_dir)
            if rel.startswith(".."):
                raise SystemExit(f"{path} is outside {work_dir}")
            arc = f"{top}/data/{rel}".replace(os.sep, "/")
            if arc not in written and os.path.exists(path):
                z.write(path, arc)
                written.add(arc)

        for name in TOOLS:
            z.write(os.path.join(HERE, name), f"{top}/{name}")
        z.writestr(f"{top}/README.md", PACKAGE_README)
        for report in reports:
            for name in REPORT_FILES:
                add(os.path.join(report, name))
            with open(os.path.join(report, "summary.json"),
                      encoding="utf-8") as f:
                summary = json.load(f)
            for entry in summary["images"]:
                add(os.path.join(report, entry["input"]))
                match_dir = os.path.normpath(
                    os.path.join(report, entry["match_dir"]))
                for name in IMAGE_FILES:
                    add(os.path.join(match_dir, name))
                arc = f"{top}/data/" + os.path.relpath(
                    os.path.join(match_dir, sp.MATCHES_NPZ),
                    work_dir).replace(os.sep, "/")
                if arc not in written:
                    buf = io.BytesIO()
                    np.savez_compressed(buf, **sp.matches_from_csv(
                        os.path.join(match_dir, "matched.csv")))
                    z.writestr(arc, buf.getvalue())
                    written.add(arc)
                    images += 1
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
