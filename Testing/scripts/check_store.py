#!/usr/bin/env python3
"""Check a result store before comparing it.

    check_store.py <store> [--tools <build>/popsift-testing-config.json]

Reads run.json and every feature file of the store, and reports:

- a manifest without 'format' or 'name', or with an unusable name,
- feature files that are not 132 numbers per line, with files in the Oxford
  affine format (133 numbers per line) named as such,
- orientations that look like degrees rather than radians,
- descriptors that look like floats in 0-1 rather than bytes in 0-255, or
  that exceed 255,
- with --tools: the images of the store's image sets (run.json
  'image_sets') that have no feature file.

Writing a store is described in Testing/scripts/README.md. The exit code is
0 when the store can be compared, 1 otherwise; warnings alone do not fail.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import result_store as rs  # noqa: E402


def feature_files(store):
    """The .sift files of a store by (set folder, image stem), skipping the
    hidden folders that collect_run.py works in."""
    found = []
    for root, dirs, files in os.walk(store.path):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for f in sorted(files):
            if f.endswith(rs.FEATURES):
                found.append(os.path.join(root, f))
    return found


def check_file(path):
    """Returns (number of keypoints, warnings). Raises TestingError when the
    file cannot be compared."""
    lines = rs.read_features(path)
    warnings = []
    if not lines:
        return 0, warnings
    values = [[float(x) for x in line.split()] for line in lines]
    oris = [v[3] for v in values]
    sigmas = [v[2] for v in values]
    desc_max = max(max(v[4:]) for v in values)
    desc_min = min(min(v[4:]) for v in values)
    if max(abs(o) for o in oris) > 2 * math.pi + 1e-3:
        warnings.append(f"orientations up to {max(oris):g}: degrees? "
                        "Expected radians")
    if min(sigmas) <= 0:
        warnings.append("a sigma of 0 or less")
    if desc_min < 0:
        warnings.append(f"negative descriptor values ({desc_min:g})")
    if desc_max > 255:
        warnings.append(f"descriptor values up to {desc_max:g}, expected "
                        "bytes 0-255")
    elif desc_max <= 1.0:
        warnings.append("all descriptor values are at most 1: floats in 0-1? "
                        "Expected bytes 0-255, as VLFeat writes them")
    return len(lines), warnings


def missing_images(store, tools):
    """The images of the store's image sets without a feature file, as
    (set, relative path); and the sets that are unknown to tools."""
    missing, unknown = [], []
    for set_name in store.manifest.get("image_sets", []):
        try:
            imgset = rs.ImageSet(tools, set_name)
            files = imgset.files()
        except rs.TestingError:
            unknown.append(set_name)
            continue
        for src in files:
            rel = imgset.rel(src)
            if not os.path.exists(store.file(set_name, rel)) \
                    and store.skipped(set_name, rel) is None:
                missing.append((set_name, rel))
    return missing, unknown


def check(path, tools=None):
    """Prints the findings. Returns the exit code."""
    try:
        store = rs.Store.open(path)
    except rs.TestingError as e:
        print(f"ERROR {e}")
        return 1
    m = store.manifest
    print(f"Store {store.name} ({store.label}) in {store.path}")
    for key in ("tool", "configuration", "version", "device", "host",
                "created"):
        if m.get(key):
            print(f"  {key}: {m[key]}")
    errors = 0
    keypoints = 0
    files = feature_files(store)
    for f in files:
        rel = os.path.relpath(f, store.path)
        try:
            count, warnings = check_file(f)
        except rs.TestingError as e:
            print(f"ERROR {e}")
            errors += 1
            continue
        keypoints += count
        for w in warnings:
            print(f"WARNING {rel}: {w}")
    print(f"{len(files)} feature file(s), {keypoints} keypoints, "
          f"{errors} unusable")
    if tools is not None:
        missing, unknown = missing_images(store, tools)
        for set_name in unknown:
            print(f"WARNING image set '{set_name}' is not configured in this "
                  "build")
        for set_name, rel in missing:
            print(f"MISSING {set_name}: {rel}")
    return 1 if errors else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    parser.add_argument("store", help="the folder of the store, or its name "
                                      "in the stores dir of --tools")
    parser.add_argument("--tools",
                        help="popsift-testing-config.json, to find the "
                             "images that have no feature file")
    args = parser.parse_args(argv)
    try:
        tools = rs.load_tools(args.tools) if args.tools else None
    except rs.TestingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    path = args.store
    if tools is not None and not os.path.isdir(path):
        path = os.path.join(rs.stores_dir(tools), path)
    return check(path, tools)


if __name__ == "__main__":
    sys.exit(main())
