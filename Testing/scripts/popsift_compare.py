#!/usr/bin/env python3
"""Compare two SIFT runs on every image of a test image set.

A run is a result store: the features of one SIFT tool with one
configuration on image sets, which collect_run.py or any other tool wrote
(see Testing/scripts/README.md). It is named by its name in the stores dir,
e.g. atan2-fast+popsift-vlfeat, or by its path. A run can also be given as a
configuration, 'vlfeat' (VLFeat's sift CLI) or a parameter set from
Testing/paramsets.json; collect_run.py then collects it into the store of
that name first, unless --no-collect is given.

For every image, the features of both stores are sorted, and
popsift-compareSiftFiles compares the first run against the second: it pairs
their keypoints by position and scale (co-located), rates every pair as
agreeing in position only, also in orientation, or also in the descriptor,
and writes colocated.csv, only_a.csv and only_b.csv, plus the descriptor
nearest neighbour of every keypoint in the other run (desc_nn_a.csv,
desc_nn_b.csv) into <work dir>/<set>/<image>/<a>__vs__<b>/. Every command
is printed, and each image ends with a one-line verdict. A report with
statistics per image and for the set (summary.csv, summary.json, report.md)
is written to <work dir>/<set>/report_<a>__vs__<b>/. With numpy and
matplotlib installed, each image and the report also get agreement.png,
cdf.png, sorted.png and descriptor_bins.png, and each image unpaired.png and
image_heatmap.png.

Nothing fails on the statistics unless limits are given (--min-recall,
--min-precision, --min-ori-agreement, --min-desc-agreement, --max-desc-dist,
--min-nn-agreement); without them an image only fails if a run crashes, or
exactly one run finds no features. A store without the image set fails; an
image that an earlier store lacks is reported and not compared.

The paths to the binaries, the parameter sets, the image sets and the
stores dir come from the popsift-testing-config.json file that CMake writes
into the build directory.

    popsift_compare.py --tools <build>/popsift-testing-config.json \\
                       --images handcrafted popsift-vlfeat vlfeat
    popsift_compare.py --tools <build>/popsift-testing-config.json \\
                       --images oxford/boat atan2-fast+popsift-vlfeat vlfeat

--image limits this to one image. --step does one step of it, which is how
the CTest tests use the script: 'run' collects one configuration on one
image (with collect_run.py), 'compare' compares two stores on one image, and
'report' writes the set report from the images compared so far.

PopSift runs need a GPU; popsift-gpu-probe tells whether there is one. The
exit code is 0 if every image passed, 77 if nothing could be done for lack
of a GPU, and 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import collect_run  # noqa: E402
import match_stats  # noqa: E402
import result_store as rs  # noqa: E402

_ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "analysis")

# Shared with collect_run.py and compare_paramsets.py.
CompareError = rs.TestingError
SKIP = collect_run.SKIP
safe_name = rs.safe_name
load_tools = rs.load_tools
find_tools_file = rs.find_tools_file
print_list = collect_run.print_list

# The line popsift-compareSiftFiles prints to stderr at the end.
AVG_DIST_RE = re.compile(r"Average best-match descriptor distance:\s*(\S+)")


def run_command(cmd, cwd, log_path):
    return collect_run.run_command(cmd, cwd, log_path)


# --- runs ---------------------------------------------------------------------

def resolve_store(tools, spec, stores):
    """The store of a run name or path. Fails if there is none."""
    store = rs.find_store(spec, stores)
    if store is None:
        hint = (" A configuration name collects the run first, unless "
                "--no-collect is given." if collect_run.is_configuration(
                    tools, spec) else "")
        raise CompareError(f"no store '{spec}': neither a folder with "
                           f"{rs.MANIFEST} nor a store in {stores}.{hint}")
    return store


def check_set(store, imgset):
    if not store.has_set(imgset.name):
        raise CompareError(f"the store {store.path} has no results for the "
                           f"image set '{imgset.name}'")


def compare_features(tools, a, b, image_dir, rel, set_name, max_avg_dist,
                     allow_empty=False):
    """Compares the features of one image in the stores a and b. Returns a
    CompareResult.

    allow_empty accepts two runs that both found no features.
    """
    outdir = clear_compare_dir(image_dir, a.name, b.name)
    print(f"-- compare {a.name} against {b.name}", flush=True)
    sorted_files = []
    for side, store in (("a", a), ("b", b)):
        dest = os.path.join(outdir, f"features_{side}{rs.FEATURES}")
        try:
            lines = rs.read_features(store.file(set_name, rel))
        except CompareError as e:
            print(f"   {e}", flush=True)
            return CompareResult(False, None, outdir, None)
        rs.write_sorted(lines, dest)
        sorted_files.append(dest)
    cmd = [tools["compare_sift_files"],
           "--max-avg-dist", str(max_avg_dist),
           "-o", os.path.join(outdir, "comparison.txt"),
           "-d", os.path.join(outdir, "descriptor-distances.txt"),
           "--pair-dir", outdir,
           "--ori-tolerance-deg", str(match_stats.ORI_TOLERANCE_DEG),
           "--desc-tolerance", str(match_stats.DESC_TOLERANCE_L2)] \
        + (["--allow-empty"] if allow_empty else []) + sorted_files
    code, output, _ = run_command(cmd, outdir,
                                  os.path.join(outdir, "log.txt"))
    match = AVG_DIST_RE.search(output)
    avg = float(match.group(1)) if match else None
    for line in output.splitlines():
        if line.startswith("FAIL"):
            print("   " + line, flush=True)
    stats = None
    if os.path.exists(os.path.join(outdir, "colocated.csv")):
        stats = match_stats.image_stats(outdir)
    return CompareResult(code == 0, avg, outdir, stats)


def availability(store, set_name, rel):
    """Whether a store has features for an image: (True, None), or
    (False, why)."""
    if os.path.exists(store.file(set_name, rel)):
        return True, None
    reason = store.skipped(set_name, rel)
    if reason is not None:
        return False, f"skipped in {store.name}: {reason}"
    return False, f"not in {store.name}"


# The tolerances of the agreement levels, for the plots.
AGREEMENT_TOLERANCES = {"ori_tolerance": match_stats.ORI_TOLERANCE_DEG,
                        "desc_tolerance": match_stats.DESC_TOLERANCE_L2}

# The files of popsift-compareSiftFiles --pair-dir.
PAIR_FILES = ("colocated.csv", "only_a.csv", "only_b.csv",
              "desc_nn_a.csv", "desc_nn_b.csv")


def compare_dir(image_dir, spec_a, spec_b):
    return os.path.join(image_dir,
                        f"{safe_name(spec_a)}__vs__{safe_name(spec_b)}")


def clear_compare_dir(image_dir, spec_a, spec_b):
    """Creates compare_dir() and removes the pair files of an earlier
    comparison, which the set report would otherwise pick up."""
    outdir = compare_dir(image_dir, spec_a, spec_b)
    os.makedirs(outdir, exist_ok=True)
    for name in PAIR_FILES:
        if os.path.exists(os.path.join(outdir, name)):
            os.remove(os.path.join(outdir, name))
    return outdir


class CompareResult:
    def __init__(self, passed, avg, pair_dir, stats):
        self.passed = passed          # the compare tool succeeded
        self.avg = avg                # average best-match descriptor distance
        self.pair_dir = pair_dir      # colocated.csv, only_a.csv, only_b.csv
        self.stats = stats            # match_stats.image_stats(), or None


def load_plotting():
    """The optional plotting module, or None without numpy and matplotlib."""
    if _ANALYSIS_DIR not in sys.path:
        sys.path.insert(0, _ANALYSIS_DIR)
    try:
        import matplotlib
        matplotlib.use("Agg")  # only files are written, never a window
        import sift_plots
    except ImportError:
        return None
    return sift_plots


# --- one image set ------------------------------------------------------------

class ImageSet(rs.ImageSet):
    """The images of a set and where their comparisons go."""

    def __init__(self, tools, images, work_dir):
        super().__init__(tools, images)
        self.dir = os.path.join(work_dir,
                                safe_name(self.name.replace("/", "_")))

    def image_dir(self, src):
        """The work directory of one image, named after it."""
        return os.path.join(self.dir,
                            safe_name(os.path.splitext(self.rel(src))[0]))

    def report_dir(self, run_a, run_b):
        return os.path.join(self.dir,
                            f"report_{safe_name(run_a)}__vs__{safe_name(run_b)}")


def compare_image(tools, imgset, src, a, b, max_avg_dist, limits, plotting):
    """Compares the stores a and b on one image and prints the verdict.

    Returns the report entry of the image (None if the comparison produced
    no statistics) and whether it passed.
    """
    rel = imgset.rel(src)
    image_dir = imgset.image_dir(src)
    pgm = collect_run.prepare_input(src, image_dir)
    res = compare_features(tools, a, b, image_dir, rel, imgset.name,
                           max_avg_dist, imgset.allow_empty(src))
    problems = []
    entry = None
    if res.stats is not None:
        problems = match_stats.check_limits(res.stats, **limits)
        entry = {"name": rel, "input": pgm, "pair_dir": res.pair_dir,
                 "stats": res.stats}
        if plotting is not None:
            plotting.render_unpaired(
                pgm, res.pair_dir,
                os.path.join(res.pair_dir, "unpaired.png"),
                label_a=a.label, label_b=b.label)
            title = f"{rel}: a = {a.label}, b = {b.label}"
            plotting.render_agreement(
                os.path.join(res.pair_dir, "agreement.png"), title,
                res.stats["_rows"], res.stats["counts"],
                **AGREEMENT_TOLERANCES)
            plotting.render_cdfs(os.path.join(res.pair_dir, "cdf.png"),
                                 title, [(rel, res.stats["_rows"])])
            render_differences(plotting, res.pair_dir, title,
                               res.stats["_rows"],
                               [plotting.read_desc_bins(res.pair_dir)])
            plotting.render_image_heatmap(
                pgm, plotting.read_pairs(res.pair_dir),
                os.path.join(res.pair_dir, "image_heatmap.png"), title)
    passed = res.passed and not problems
    verdict = "PASSED" if passed else "FAILED"
    detail = match_stats.one_line(res.stats) if res.stats else \
        "no statistics"
    print(f"{verdict} {rel}: {detail}")
    for p in problems:
        print(f"   limit: {p}")
    if not passed:
        print(f"   details: {res.pair_dir}")
    return entry, passed


def render_differences(plotting, out_dir, title, rows, desc_bins):
    """Writes sorted.png and, if desc_bins.csv exists, descriptor_bins.png
    (the plots of the 2021 comparison scripts) into out_dir."""
    plotting.render_sorted(os.path.join(out_dir, "sorted.png"), title, rows)
    bins = plotting.merge_desc_bins(desc_bins)
    if bins:
        plotting.render_desc_bins(
            os.path.join(out_dir, "descriptor_bins.png"), title, bins)


def write_set_report(imgset, run_a, run_b, reported, plotting, runs=None):
    """Writes the report of the set over the reported images. runs holds
    the manifests of the stores a and b, if known."""
    report_dir = imgset.report_dir(run_a, run_b)
    total = match_stats.set_stats([r["stats"] for r in reported])
    if plotting is not None:
        os.makedirs(report_dir, exist_ok=True)
        title = (f"{imgset.name}, {len(reported)} image(s): a = {run_a}, "
                 f"b = {run_b}")
        plotting.render_agreement(os.path.join(report_dir, "agreement.png"),
                                  title, total["_rows"], total["counts"],
                                  **AGREEMENT_TOLERANCES)
        plotting.render_cdfs(os.path.join(report_dir, "cdf.png"), title,
                             [(imgset.name, total["_rows"])])
        render_differences(plotting, report_dir, title, total["_rows"],
                           [plotting.read_desc_bins(r["pair_dir"])
                            for r in reported])
    match_stats.write_report(report_dir, run_a, run_b, imgset.name, reported,
                             total, runs)
    print(f"Set: {match_stats.one_line(total)}")
    print(f"Report: {report_dir}/report.md (also summary.csv, summary.json)")


def plotting_or_note(plots):
    plotting = load_plotting() if plots else None
    if plots and plotting is None:
        print("Note: install numpy and matplotlib (Testing/analysis/"
              "requirements.txt) to get the plots.")
    return plotting


def manifests(a, b):
    return {"a": a.manifest, "b": b.manifest}


def collect_or_open(tools, specs, images, stores, collect, only):
    """The stores of the runs specs. A configuration name ('vlfeat' or a
    parameter set) is collected by collect_run.py first, unless collect is
    False. Returns the stores and the names of those collected now."""
    found = {}
    fresh = set()
    for spec in dict.fromkeys(specs):
        if collect and collect_run.is_configuration(tools, spec):
            collector = collect_run.Collector(tools, spec, stores=stores,
                                              replace=True)
            collector.collect(images, only)
            found[spec] = rs.Store.open(collector.path)
            fresh.add(found[spec].name)
        else:
            found[spec] = resolve_store(tools, spec, stores)
    return [found[s] for s in specs], fresh


def compare_one_image(tools, imgset, src, a, b, fresh, max_avg_dist,
                      limits, plotting):
    """Compares a and b on one image of the set. Returns its outcome,
    'passed', 'failed', 'skipped' (no GPU) or 'missing' (not in an earlier
    store), and its report entry or None."""
    rel = imgset.rel(src)
    why = [w for ok, w in (availability(s, imgset.name, rel)
                           for s in (a, b)) if not ok]
    if any(w.startswith("skipped") for w in why):
        print(f"SKIPPED {rel}: {'; '.join(why)}")
        return "skipped", None
    if why:
        # A run of this invocation that left no features failed; an
        # image missing from an earlier store is only reported.
        if {a.name, b.name} & fresh:
            print(f"FAILED {rel}: a run did not produce features "
                  f"({'; '.join(why)})")
            return "failed", None
        print(f"NOT COMPARED {rel}: {'; '.join(why)}")
        return "missing", None
    try:
        entry, passed = compare_image(tools, imgset, src, a, b,
                                      max_avg_dist, limits, plotting)
    except CompareError as e:
        print(f"FAILED {rel}: {e}")
        return "failed", None
    return ("passed" if passed else "failed"), entry


def print_set_summary(imgset, count, outcomes):
    """Prints how the images of a set went. outcomes maps 'failed',
    'skipped' and 'missing' to the images. Returns the exit code."""
    failed, skipped, missing = (outcomes[k] for k in
                                ("failed", "skipped", "missing"))
    done = count - len(skipped) - len(missing)
    if done:
        print(f"{done - len(failed)} of {done} image(s) passed"
              + (f", {len(skipped)} skipped for lack of a GPU" if skipped
                 else "")
              + (f", {len(missing)} not in the stores" if missing else ""))
    elif skipped:
        print(f"All {len(skipped)} image(s) skipped for lack of a GPU")
    for rel in failed:
        print(f"  failed: {rel}")
    for rel in missing:
        print(f"  not in the stores: {rel}")
    if failed:
        print(f"Details are in the per-image folders under {imgset.dir}")
        return 1
    if not done:
        return SKIP if skipped else 1
    return 0


def compare_image_set(tools, images, run_a, run_b, work_dir, max_avg_dist,
                      limits=None, plots=True, only=None, stores=None,
                      collect=True):
    """Compares a and b on every image of a set, or only on the image named
    only, and writes the set report. Runs given as configurations are
    collected first."""
    stores = rs.stores_dir(tools, stores)
    imgset = ImageSet(tools, images, work_dir)
    collect_run.check_binaries(tools, ["compare_sift_files"])
    files = [imgset.find(only)] if only else imgset.files()
    (a, b), fresh = collect_or_open(tools, [run_a, run_b], images, stores,
                                    collect, only)
    for store in (a, b):
        check_set(store, imgset)
    print(f"\nComparing {a.name} against {b.name} on {len(files)} image(s) "
          f"of '{imgset.name}' in {imgset.folder}")
    print(f"Stores: {a.path}, {b.path}")
    print(f"Work directory: {imgset.dir}")
    plotting = plotting_or_note(plots)

    outcomes = {"passed": [], "failed": [], "skipped": [], "missing": []}
    reported = []
    for index, src in enumerate(files, 1):
        rel = imgset.rel(src)
        print(f"\n==================== [{index}/{len(files)}] {rel} "
              "====================", flush=True)
        outcome, entry = compare_one_image(tools, imgset, src, a, b, fresh,
                                           max_avg_dist, limits or {},
                                           plotting)
        outcomes[outcome].append(rel)
        if entry is not None:
            reported.append(entry)

    print("\n==================== Summary ====================")
    if reported:
        write_set_report(imgset, a.name, b.name, reported, plotting,
                         manifests(a, b))
    return print_set_summary(imgset, len(files), outcomes)


# --- single steps, one CTest test each ----------------------------------------

def step_run(tools, images, key, spec, stores):
    """Runs one configuration on one image, with collect_run.py."""
    collector = collect_run.Collector(tools, spec, stores=stores,
                                      replace=True)
    return collect_run.summarize(collector.collect(images, key))


def step_compare(tools, images, key, run_a, run_b, work_dir, max_avg_dist,
                 limits, plots, stores):
    """Compares the stores a and b on one image."""
    stores = rs.stores_dir(tools, stores)
    imgset = ImageSet(tools, images, work_dir)
    collect_run.check_binaries(tools, ["compare_sift_files"])
    src = imgset.find(key)
    rel = imgset.rel(src)
    a, b = (resolve_store(tools, s, stores) for s in (run_a, run_b))
    print(f"Comparing {a.name} against {b.name} on {rel} of "
          f"'{imgset.name}'")
    clear_compare_dir(imgset.image_dir(src), a.name, b.name)
    for store in (a, b):
        ok, why = availability(store, imgset.name, rel)
        if not ok and why.startswith("skipped"):
            print(f"SKIPPED {rel}: {why}")
            return SKIP
        if not ok:
            print(f"FAILED {rel}: {why}; the run failed or did not run")
            return 1
    try:
        _, passed = compare_image(tools, imgset, src, a, b, max_avg_dist,
                                  limits, plotting_or_note(plots))
    except CompareError as e:
        print(f"FAILED {rel}: {e}")
        return 1
    return 0 if passed else 1


def step_report(tools, images, run_a, run_b, work_dir, plots, stores):
    """Writes the set report from the image comparisons that exist."""
    stores = rs.stores_dir(tools, stores)
    imgset = ImageSet(tools, images, work_dir)
    files = rs.find_images(imgset.folder)
    if not files:
        # The images were removed after configure.
        print(f"SKIPPED: no images in {imgset.folder}")
        return SKIP
    found = [rs.find_store(s, stores) for s in (run_a, run_b)]
    names = [f.name if f else safe_name(s) for f, s in zip(found, (run_a,
                                                                   run_b))]
    reported = []
    missing = []
    for src in files:
        image_dir = imgset.image_dir(src)
        pair_dir = compare_dir(image_dir, *names)
        if not os.path.exists(os.path.join(pair_dir, "colocated.csv")):
            missing.append(imgset.rel(src))
            continue
        reported.append({"name": imgset.rel(src),
                         "input": collect_run.input_path(src, image_dir),
                         "pair_dir": pair_dir,
                         "stats": match_stats.image_stats(pair_dir)})
    print(f"Report of {names[0]} against {names[1]} on '{imgset.name}': "
          f"{len(reported)} compared image(s)")
    for rel in missing:
        print(f"  not compared: {rel}")
    if not reported:
        print("SKIPPED: no image of the set was compared")
        return SKIP
    runs = manifests(*found) if all(found) else None
    write_set_report(imgset, *names, reported, plotting_or_note(plots), runs)
    return 0


def make_parser():
    """The command line parser of the script."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Runs are result stores (a name in the stores dir or a "
               "path), or configurations: 'vlfeat' or a parameter set from "
               "Testing/paramsets.json, optionally with overrides such as "
               "loop@gauss-mode=vlfeat, which collect_run.py collects first.")
    parser.add_argument("--tools", required=True,
                        help="popsift-testing-config.json from the build directory")
    parser.add_argument("--list", action="store_true",
                        help="list the parameter sets and image sets, and exit")
    parser.add_argument("--images",
                        help="a configured image set (e.g. handcrafted, "
                             "oxford/boat), the start of the labels of "
                             "several (oxford, compared one after the "
                             "other), or a folder of .pgm/.ppm files")
    parser.add_argument("--image",
                        help="only this image of the set: its path in the "
                             "set, with or without extension")
    parser.add_argument("--step", choices=("run", "compare", "report"),
                        help="do one step only: 'run' one configuration on "
                             "--image (with collect_run.py), 'compare' the "
                             "stores a and b on --image, or 'report' on the "
                             "compared images of the set. Default: all of "
                             "them")
    parser.add_argument("--stores",
                        help="where the result stores are (default: "
                             "stores_dir from --tools)")
    parser.add_argument("--no-collect", action="store_true",
                        help="use the existing stores of configurations "
                             "instead of collecting them again")
    parser.add_argument("--work-dir",
                        help="where outputs go (default: work_dir from --tools)")
    parser.add_argument("--max-avg-dist", type=float, default=-1.0,
                        help="fail an image whose average best-match descriptor "
                             "distance exceeds this; negative only reports "
                             "(default: %(default)s)")
    parser.add_argument("--min-recall", type=float, default=-1.0,
                        help="fail an image where fewer than this share of "
                             "run_b's keypoints are co-located with one of "
                             "run_a (negative: off)")
    parser.add_argument("--min-precision", type=float, default=-1.0,
                        help="fail an image where fewer than this share of "
                             "run_a's keypoints are co-located with one of "
                             "run_b (negative: off)")
    parser.add_argument("--min-ori-agreement", type=float, default=-1.0,
                        help="fail an image where fewer than this share of "
                             "the co-located pairs have the same orientation "
                             "(negative: off)")
    parser.add_argument("--min-desc-agreement", type=float, default=-1.0,
                        help="fail an image where fewer than this share of "
                             "the pairs with the same orientation have a "
                             "similar descriptor (negative: off)")
    parser.add_argument("--max-desc-dist", type=float, default=-1.0,
                        help="fail an image whose median descriptor distance "
                             "of the pairs with the same orientation exceeds "
                             "this (negative: off)")
    parser.add_argument("--min-nn-agreement", type=float, default=-1.0,
                        help="fail an image where fewer than this share of "
                             "the co-located keypoints of either run have "
                             "their co-located keypoint as descriptor "
                             "nearest neighbour (negative: off)")
    parser.add_argument("--no-plots", action="store_true",
                        help="do not write the plots")
    parser.add_argument("runs", nargs="*", metavar="run",
                        help="the run under test (a) and the reference it is "
                             "compared with (b); only one run with --step run")
    return parser


def check_args(parser, args):
    """Checks the arguments that argparse cannot check by itself."""
    if not args.images:
        parser.error("--images is required (or use --list)")
    wanted = 1 if args.step == "run" else 2
    if len(args.runs) != wanted:
        got = ", ".join(f"'{r}'" for r in args.runs) or "none"
        parser.error(
            f"expected {wanted} run(s) at the end of the command, got "
            f"{len(args.runs)}: {got}. "
            + ("The run under test (a) comes first, the reference (b) "
               "second; each is a store (a name in the stores dir or a "
               "path) or a configuration. " if wanted == 2 else "")
            + "Options such as --stores, --work-dir and --images take a "
              "value of their own, so a run given right after one of them "
              "is taken as its value.")
    if args.step in ("run", "compare") and not args.image:
        parser.error(f"--step {args.step} needs --image")


def limits_of(args):
    """The pass/fail limits of match_stats.check_limits() from args."""
    return {"min_recall": args.min_recall,
            "min_precision": args.min_precision,
            "min_ori_agreement": args.min_ori_agreement,
            "min_desc_agreement": args.min_desc_agreement,
            "max_desc_dist": args.max_desc_dist,
            "min_nn_agreement": args.min_nn_agreement}


def run_step(tools, images, args, work_dir):
    """Does the single step args.step on one image set."""
    plots = not args.no_plots
    if args.step == "run":
        return step_run(tools, images, args.image, args.runs[0],
                        args.stores)
    if args.step == "compare":
        return step_compare(tools, images, args.image, *args.runs, work_dir,
                            args.max_avg_dist, limits_of(args), plots,
                            args.stores)
    return step_report(tools, images, *args.runs, work_dir, plots,
                       args.stores)


def compare_sets(tools, image_sets, args, work_dir):
    """Compares the runs on every image set, one after the other, and
    returns the combined exit code."""
    codes = [compare_image_set(tools, images, *args.runs, work_dir,
                               args.max_avg_dist, limits_of(args),
                               not args.no_plots, args.image, args.stores,
                               not args.no_collect)
             for images in image_sets]
    if len(image_sets) > 1:
        print("\n==================== All sets ====================")
        for images, code in zip(image_sets, codes):
            verdict = {0: "PASSED", SKIP: "SKIPPED"}.get(code, "FAILED")
            print(f"{verdict} {images}")
    if any(c not in (0, SKIP) for c in codes):
        return 1
    return SKIP if all(c == SKIP for c in codes) else 0


def main(argv=None):
    parser = make_parser()
    args = parser.parse_intermixed_args(argv)
    try:
        if args.list:
            print_list(load_tools(args.tools))
            return 0
        check_args(parser, args)
        tools = load_tools(args.tools)
        image_sets = list(dict.fromkeys(rs.expand_images(tools,
                                                         args.images)))
        if (args.step or args.image) and len(image_sets) != 1:
            parser.error(f"--step and --image need a single image set, not "
                         f"{', '.join(image_sets)}")
        work_dir = args.work_dir or tools.get("work_dir") or os.getcwd()
        if args.step:
            return run_step(tools, image_sets[0], args, work_dir)
        return compare_sets(tools, image_sets, args, work_dir)
    except CompareError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
